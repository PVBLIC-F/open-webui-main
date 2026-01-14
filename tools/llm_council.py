"""
title: LLM Council Tool
author: Chris Headings (refactored for Open WebUI)
version: 1.5.0
license: MIT
description: 3-stage LLM Council with consensus analysis and optional Exa fact verification - multiple models deliberate to produce comprehensive answers.
required_open_webui_version: 0.7.2
requirements: aiohttp
"""

import asyncio
import aiohttp
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

DEFAULT_COUNCIL_MODELS = "anthropic/claude-opus-4.5,openai/gpt-4o,google/gemini-2.0-flash-001"


class Tools:
    """LLM Council Tool - multi-model deliberation with consensus analysis."""

    class Valves(BaseModel):
        api_key: str = Field(
            default="",
            description="⚠️ REQUIRED: Your Open WebUI API key (Settings → Account → API Keys)",
        )
        council_models: str = Field(
            default=DEFAULT_COUNCIL_MODELS,
            description="Comma-separated model IDs",
        )
        chairperson_model: str = Field(
            default="",
            description="Chairperson model (empty = first council model)",
        )
        max_models: int = Field(default=5, description="Max models")
        timeout: int = Field(default=180, description="Timeout per model (seconds)")
        show_consensus_analysis: bool = Field(default=True, description="Show agreement/disagreement analysis")
        show_individual_responses: bool = Field(default=True)
        show_peer_rankings: bool = Field(default=True)
        show_aggregate_rankings: bool = Field(default=True)
        # Exa Fact Verification
        exa_api_key: str = Field(
            default="",
            description="Exa API key for fact verification (optional - get one at exa.ai)",
        )
        enable_fact_verification: bool = Field(
            default=False,
            description="Enable fact verification using Exa web search",
        )
        max_claims_to_verify: int = Field(
            default=5,
            description="Maximum number of claims to verify per query (to control costs)",
        )
        show_fact_verification: bool = Field(
            default=True,
            description="Show fact verification results in report",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_models(self) -> list[str]:
        """Parse council models, stripping quotes."""
        raw = self.valves.council_models.strip().strip("'\"")
        models = [m.strip().strip("'\"") for m in raw.split(",") if m.strip()]
        return models[:self.valves.max_models]

    async def _emit_status(self, emitter, msg: str, done: bool = False):
        if emitter and callable(emitter):
            try:
                await emitter({"type": "status", "data": {"description": msg, "done": done}})
            except:
                pass

    async def _call_model(self, model: str, messages: list, base_url: str, api_key: str) -> Optional[str]:
        """Call model via HTTP API."""
        url = f"{base_url}/api/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}

        log.debug(f"  📤 {model}")
        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        log.error(f"  ❌ {model}: HTTP {resp.status} - {err[:100]}")
                        return None
                    data = await resp.json()
                    if "choices" in data and data["choices"]:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content:
                            log.debug(f"  ✅ {model}: {len(content)} chars")
                            return content
                    log.warning(f"  ⚠️ {model}: No content")
                    return None
        except asyncio.TimeoutError:
            log.error(f"  ❌ {model}: Timeout ({self.valves.timeout}s)")
            return None
        except Exception as e:
            log.error(f"  ❌ {model}: {e}")
            return None

    async def _call_parallel(self, models: list, messages: list, base_url: str, api_key: str) -> dict:
        """Call models in parallel."""
        log.info(f"🔄 Calling {len(models)} models...")
        
        async def call_one(m):
            return m, await self._call_model(m, messages, base_url, api_key)
        
        results = await asyncio.gather(*[call_one(m) for m in models])
        responses = {m: r for m, r in results}
        success = sum(1 for r in responses.values() if r)
        log.info(f"🔄 {success}/{len(models)} succeeded")
        return responses

    def _parse_ranking(self, text: str) -> list[str]:
        if not text:
            return []
        upper = text.upper()
        if "FINAL RANKING:" in upper:
            section = upper.split("FINAL RANKING:")[1]
            matches = re.findall(r"Response\s+([A-Z])", section)
            if matches:
                return [f"Response {m}" for m in matches]
        matches = re.findall(r"Response\s+([A-Z])", upper)
        return [f"Response {m}" for m in dict.fromkeys(matches)]

    def _calc_rankings(self, stage2: list, label_map: dict) -> list:
        positions = defaultdict(list)
        for r in stage2:
            for pos, label in enumerate(r.get("parsed_ranking", []), 1):
                if label in label_map:
                    positions[label_map[label]].append(pos)
        return sorted([
            {"model": m, "avg": round(sum(p)/len(p), 2), "votes": len(p)}
            for m, p in positions.items() if p
        ], key=lambda x: x["avg"])

    # ═══════════════════════════════════════════════════════════════════
    # Exa Fact Verification
    # ═══════════════════════════════════════════════════════════════════

    async def _extract_claims(
        self,
        topic: str,
        stage1: list[dict],
        chair: str,
        base_url: str,
        api_key: str,
    ) -> list[str]:
        """Extract verifiable factual claims from Stage 1 responses."""
        responses_text = "\n\n".join([
            f"**{s['model']}:**\n{s['response']}" for s in stage1
        ])

        extract_prompt = f"""Analyze these AI responses to: "{topic}"

{responses_text}

Extract ALL FACTUAL CLAIMS that can be verified with a web search.

Rules:
- Extract SPECIFIC, VERIFIABLE facts (dates, numbers, names, events, statistics, quotes)
- Skip opinions, recommendations, or general statements
- Combine similar claims from different models into one
- Extract as many claims as possible, up to {self.valves.max_claims_to_verify} claims
- Format each claim as a complete, searchable question

IMPORTANT: Try to extract close to {self.valves.max_claims_to_verify} claims if there are enough verifiable facts.

Return ONLY a JSON array of strings, nothing else:
["Is it true that X?", "Did Y happen in Z?", ...]

If there are no verifiable factual claims, return: []"""

        result = await self._call_model(chair, [{"role": "user", "content": extract_prompt}], base_url, api_key)
        
        if not result:
            return []
        
        # Parse JSON from response
        try:
            # Try to find JSON array in response
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                claims = json.loads(match.group())
                if isinstance(claims, list):
                    return claims[:self.valves.max_claims_to_verify]
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse claims JSON: {e}")
        
        return []

    async def _verify_claim_with_exa(
        self, 
        claim: str, 
        semaphore: asyncio.Semaphore,
        max_retries: int = 3
    ) -> dict:
        """Verify a single claim using Exa's /answer API with rate limiting and retry."""
        url = "https://api.exa.ai/answer"
        headers = {
            "x-api-key": self.valves.exa_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "query": claim,
            "text": True,
        }

        async with semaphore:  # Limit concurrent requests
            for attempt in range(max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, json=payload, headers=headers) as resp:
                            if resp.status == 429:
                                # Rate limited - wait and retry
                                retry_after = int(resp.headers.get("Retry-After", 2))
                                wait_time = retry_after * (attempt + 1)  # Exponential backoff
                                log.warning(f"Exa rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                            
                            if resp.status != 200:
                                err = await resp.text()
                                log.error(f"Exa API error: HTTP {resp.status} - {err[:100]}")
                                return {"claim": claim, "status": "error", "answer": f"API error: {resp.status}"}
                            
                            data = await resp.json()
                            answer = data.get("answer", "")
                            citations = data.get("citations", [])
                            
                            # Extract source URLs
                            sources = [c.get("url", "") for c in citations[:3] if c.get("url")]
                            
                            # Small delay after successful request to avoid rate limits
                            await asyncio.sleep(0.5)
                            
                            return {
                                "claim": claim,
                                "status": "verified",
                                "answer": answer,
                                "sources": sources,
                            }
                except asyncio.TimeoutError:
                    log.error(f"Exa API timeout for claim: {claim[:50]}...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return {"claim": claim, "status": "timeout", "answer": "Request timed out"}
                except Exception as e:
                    log.error(f"Exa API error: {e}")
                    return {"claim": claim, "status": "error", "answer": str(e)}
            
            # All retries exhausted
            return {"claim": claim, "status": "error", "answer": "Rate limit exceeded after retries"}

    async def _verify_claims(self, claims: list[str]) -> list[dict]:
        """Verify multiple claims with rate limiting using Exa."""
        if not claims:
            return []
        
        log.info(f"🔍 Verifying {len(claims)} claims with Exa (rate-limited)...")
        
        # Limit to 2 concurrent requests to avoid rate limiting
        semaphore = asyncio.Semaphore(2)
        
        tasks = [self._verify_claim_with_exa(claim, semaphore) for claim in claims]
        results = await asyncio.gather(*tasks)
        
        success = sum(1 for r in results if r.get("status") == "verified")
        log.info(f"🔍 Verified {success}/{len(claims)} claims")
        
        return results

    async def _analyze_consensus(
        self,
        topic: str,
        stage1: list[dict],
        chair: str,
        base_url: str,
        api_key: str,
    ) -> Optional[str]:
        """Analyze consensus and disagreement across responses."""
        responses_text = "\n\n".join([
            f"**{s['model']}:**\n{s['response']}" for s in stage1
        ])

        consensus_prompt = f"""Analyze the consensus among these AI model responses to: "{topic}"

{responses_text}

Provide a structured analysis with these exact sections:

## ✅ Strong Agreement
List key points where ALL or MOST models agree. Be specific about what they agree on.

## ⚠️ Points of Disagreement  
List any contradictions or conflicting information between models. Note which models disagree and on what.

## 💡 Unique Insights
List interesting points that only ONE model mentioned that others missed.

## 📊 Confidence Assessment
Rate overall consensus: HIGH (strong agreement), MEDIUM (some disagreement), or LOW (significant conflicts).
Briefly explain your assessment.

Be concise but thorough. Use bullet points."""

        return await self._call_model(chair, [{"role": "user", "content": consensus_prompt}], base_url, api_key)

    async def consult_council(
        self,
        topic: str,
        __user__: dict = {},
        __event_emitter__: Any = None,
        __request__: Any = None,
    ) -> str:
        """
        Convene an LLM Council to deliberate on a topic.

        Multiple AI models discuss the topic, analyze consensus, rank each other, 
        and a chairperson synthesizes the final answer.

        :param topic: The question or topic for the council
        :return: Council report with consensus analysis and synthesis
        """
        log.info("=" * 50)
        log.info("🏛️ LLM COUNCIL STARTING")
        log.info(f"Topic: {topic[:80]}...")

        # Get API key
        api_key = self.valves.api_key.strip()
        if not api_key:
            return """❌ **API Key Required**

To use the LLM Council, you need to set your API key:

1. Go to **Settings → Account → API Keys**
2. Create a new API key
3. Click the ⚙️ gear icon on this tool
4. Paste your API key in the `api_key` field

This is needed because the council makes multiple model calls with longer timeouts than the default socket system allows."""

        # Get base URL
        base_url = os.environ.get("WEBUI_URL", "http://localhost:8080")
        if __request__:
            try:
                if hasattr(__request__, "base_url"):
                    base_url = str(__request__.base_url).rstrip("/")
            except:
                pass
        
        log.info(f"📡 API: {base_url}")

        models = self._get_models()
        if not models:
            return "Error: No council models configured."
        
        chair = self.valves.chairperson_model.strip().strip("'\"") or models[0]
        log.info(f"📋 Council: {models}")
        log.info(f"👑 Chair: {chair}")

        await self._emit_status(__event_emitter__, f"Council: {', '.join(models)}")

        # ═══════════════════════════════════════════════════════════════════
        # Stage 1: Individual responses
        # ═══════════════════════════════════════════════════════════════════
        log.info("-" * 40)
        log.info("📝 STAGE 1: Individual Responses")
        await self._emit_status(__event_emitter__, f"Stage 1: Querying {len(models)} models...")
        
        responses = await self._call_parallel(models, [{"role": "user", "content": topic}], base_url, api_key)
        stage1 = [{"model": m, "response": r} for m, r in responses.items() if r]

        if not stage1:
            await self._emit_status(__event_emitter__, "Failed", done=True)
            return "Error: No models responded. Check your API key and model IDs."

        log.info(f"📝 Stage 1: {len(stage1)}/{len(models)} responses")

        # ═══════════════════════════════════════════════════════════════════
        # Stage 1.5a: Fact Verification with Exa (optional)
        # ═══════════════════════════════════════════════════════════════════
        fact_verification = []
        if self.valves.enable_fact_verification and self.valves.exa_api_key:
            log.info("-" * 40)
            log.info("🔍 FACT VERIFICATION (Exa)")
            await self._emit_status(__event_emitter__, "Extracting claims...")
            
            # Extract factual claims from responses
            claims = await self._extract_claims(topic, stage1, chair, base_url, api_key)
            log.info(f"🔍 Extracted {len(claims)} claims to verify")
            
            if claims:
                await self._emit_status(__event_emitter__, f"Verifying {len(claims)} claims with Exa...")
                fact_verification = await self._verify_claims(claims)
                log.info(f"🔍 Fact verification complete: {len(fact_verification)} results")
        elif self.valves.enable_fact_verification and not self.valves.exa_api_key:
            log.warning("Fact verification enabled but no Exa API key provided")

        # ═══════════════════════════════════════════════════════════════════
        # Stage 1.5b: Consensus Analysis (optional)
        # ═══════════════════════════════════════════════════════════════════
        consensus_analysis = None
        if self.valves.show_consensus_analysis and len(stage1) > 1:
            log.info("-" * 40)
            log.info("🔍 CONSENSUS ANALYSIS")
            await self._emit_status(__event_emitter__, "Analyzing consensus...")
            
            consensus_analysis = await self._analyze_consensus(topic, stage1, chair, base_url, api_key)
            log.info(f"🔍 Consensus analysis: {'✅' if consensus_analysis else '❌'}")

        # ═══════════════════════════════════════════════════════════════════
        # Stage 2: Peer ranking
        # ═══════════════════════════════════════════════════════════════════
        log.info("-" * 40)
        log.info("🗳️ STAGE 2: Peer Rankings")
        await self._emit_status(__event_emitter__, "Stage 2: Peer evaluation...")

        labels = [chr(65 + i) for i in range(len(stage1))]
        label_map = {f"Response {l}": s["model"] for l, s in zip(labels, stage1)}
        
        rank_prompt = f"""Evaluate responses to: {topic}

{chr(10).join([f"Response {l}:{chr(10)}{s['response']}" for l, s in zip(labels, stage1)])}

Rate each, then:

FINAL RANKING:
1. Response X
2. Response Y"""

        rank_resp = await self._call_parallel(models, [{"role": "user", "content": rank_prompt}], base_url, api_key)
        stage2 = [{"model": m, "ranking": r, "parsed_ranking": self._parse_ranking(r)} for m, r in rank_resp.items() if r]
        rankings = self._calc_rankings(stage2, label_map)

        # ═══════════════════════════════════════════════════════════════════
        # Stage 3: Synthesis
        # ═══════════════════════════════════════════════════════════════════
        log.info("-" * 40)
        log.info("👑 STAGE 3: Final Synthesis")
        await self._emit_status(__event_emitter__, f"Stage 3: {chair} synthesizing...")

        # Include consensus analysis in synthesis prompt if available
        consensus_context = ""
        if consensus_analysis:
            consensus_context = f"""
Consensus Analysis:
{consensus_analysis}

"""

        # Include fact verification in synthesis prompt if available
        verification_context = ""
        if fact_verification:
            verified_items = []
            for v in fact_verification:
                status_icon = "✅" if v.get("status") == "verified" else "⚠️"
                verified_items.append(f"- {status_icon} **{v.get('claim', '')}**\n  Answer: {v.get('answer', 'N/A')}")
            verification_context = f"""
Fact Verification (from Exa web search):
{chr(10).join(verified_items)}

IMPORTANT: Use these verified facts to correct any inaccuracies in the model responses.
"""

        synth_prompt = f"""Synthesize council responses to: {topic}

Responses:
{chr(10).join([f"**{s['model']}**: {s['response']}" for s in stage1])}

{consensus_context}{verification_context}Rankings:
{chr(10).join([f"{s['model']}: {' → '.join(s.get('parsed_ranking', []))}" for s in stage2])}

Provide a comprehensive final answer that:
1. Emphasizes points of strong agreement
2. Addresses any disagreements or uncertainties
3. Incorporates unique valuable insights
4. Uses verified facts from web search where available
5. Is well-structured and actionable"""

        synthesis = await self._call_model(chair, [{"role": "user", "content": synth_prompt}], base_url, api_key)
        
        await self._emit_status(__event_emitter__, "Complete", done=True)
        log.info("🏛️ COMPLETE")

        # ═══════════════════════════════════════════════════════════════════
        # Build Report
        # ═══════════════════════════════════════════════════════════════════
        model_to_label = {v: k for k, v in label_map.items()}
        parts = [
            f"# 🏛️ LLM Council Report\n",
            f"**Topic:** {topic}\n",
            f"**Council:** {', '.join(models)}\n",
            f"**Chairperson:** {chair}\n",
            "---\n",
            "## 📜 Final Synthesis\n",
            f"*By {chair}*\n\n",
            synthesis or "Synthesis failed.",
        ]

        # Consensus Analysis section
        if self.valves.show_consensus_analysis and consensus_analysis:
            parts.append("\n\n---\n## 🔍 Consensus Analysis\n")
            parts.append(consensus_analysis)

        # Fact Verification section
        if self.valves.show_fact_verification and fact_verification:
            parts.append("\n\n---\n## ✅ Fact Verification (Exa)\n")
            parts.append("*Claims verified against current web sources*\n\n")
            parts.append("| Claim | Verification | Sources |")
            parts.append("|-------|--------------|---------|")
            for v in fact_verification:
                # Truncate and escape special chars to avoid breaking markdown table
                raw_claim = v.get("claim", "").replace("\n", " ").strip()
                raw_answer = v.get("answer", "N/A").replace("\n", " ").strip()
                claim = (raw_claim[:60] + "..." if len(raw_claim) > 60 else raw_claim).replace("|", "\\|")
                answer = (raw_answer[:80] + "..." if len(raw_answer) > 80 else raw_answer).replace("|", "\\|")
                sources = v.get("sources", [])
                if sources:
                    source_links = ", ".join([f"[{i+1}]({url})" for i, url in enumerate(sources[:2])])
                else:
                    source_links = "—"
                status_icon = "✅" if v.get("status") == "verified" else "⚠️" if v.get("status") == "timeout" else "❌"
                parts.append(f"| {status_icon} {claim} | {answer} | {source_links} |")

        # Aggregate Rankings
        if self.valves.show_aggregate_rankings and rankings:
            parts.append("\n\n---\n## 📊 Model Rankings\n")
            parts.append("*Based on peer evaluation*\n\n")
            parts.append("| Rank | Model | Avg Position | Votes |")
            parts.append("|------|-------|--------------|-------|")
            for i, r in enumerate(rankings, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
                parts.append(f"| {medal} | {r['model']} | {r['avg']} | {r['votes']} |")

        # Individual Responses (collapsible)
        if self.valves.show_individual_responses:
            parts.append("\n\n---\n## 💭 Individual Responses\n")
            for s in stage1:
                label = model_to_label.get(s["model"], "")
                parts.append(f"\n<details>\n<summary><strong>{s['model']}</strong> {label}</summary>\n\n{s['response']}\n\n</details>")

        # Peer Rankings (collapsible)
        if self.valves.show_peer_rankings and stage2:
            parts.append("\n\n---\n## 🗳️ Peer Evaluations\n")
            for s in stage2:
                parsed = s.get("parsed_ranking", [])
                ranking_str = " → ".join(parsed) if parsed else "N/A"
                full_eval = s.get("ranking", "")
                parts.append(f"\n<details>\n<summary><strong>{s['model']}</strong>: {ranking_str}</summary>\n\n{full_eval}\n\n</details>")

        return "\n".join(parts)
