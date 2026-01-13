"""
title: LLM Council Tool
author: matheusbuniotto (refactored for Open WebUI)
funding_url: https://github.com/matheusbuniotto/openwebui-tools
version: 1.3.0
license: MIT
description: 3-stage LLM Council with consensus analysis - multiple models deliberate to produce comprehensive answers.
required_open_webui_version: 0.3.0
requirements: aiohttp
"""

import asyncio
import aiohttp
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
        # Stage 1.5: Consensus Analysis (optional)
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

        synth_prompt = f"""Synthesize council responses to: {topic}

Responses:
{chr(10).join([f"**{s['model']}**: {s['response']}" for s in stage1])}

{consensus_context}Rankings:
{chr(10).join([f"{s['model']}: {' → '.join(s.get('parsed_ranking', []))}" for s in stage2])}

Provide a comprehensive final answer that:
1. Emphasizes points of strong agreement
2. Addresses any disagreements or uncertainties
3. Incorporates unique valuable insights
4. Is well-structured and actionable"""

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
