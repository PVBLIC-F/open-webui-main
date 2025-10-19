"""
Email Content Enricher

Advanced email content analysis and metadata enrichment for improved
semantic search and content discovery.

Features:
- Named Entity Recognition (people, organizations, locations, dates)
- Topic classification and sentiment analysis
- Key phrase extraction and semantic tagging
- Content type classification
- Relationship extraction
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class EntityInfo:
    """Information about extracted entities"""
    type: str  # 'person', 'organization', 'location', 'date', 'money', etc.
    value: str
    confidence: float
    context: str


@dataclass
class ContentAnalysis:
    """Complete content analysis results"""
    entities: Dict[str, List[EntityInfo]]
    topics: List[str]
    sentiment: Dict[str, float]
    key_phrases: List[str]
    semantic_tags: List[str]
    content_type: str
    action_items: List[str]
    relationships: List[Dict]


class EmailContentEnricher:
    """
    Advanced email content analysis and enrichment for semantic search.
    
    Provides comprehensive content understanding including entity extraction,
    topic classification, sentiment analysis, and relationship mapping.
    """

    def __init__(self):
        # Entity extraction patterns - precise patterns only
        self.entity_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+1\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US phone numbers
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'money': r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # Proper money format: $1,000.00
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'date': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            'time': r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b',  # Require AM/PM
        }

        # Topic keywords for classification
        self.topic_keywords = {
            'project_management': ['project', 'milestone', 'deadline', 'task', 'deliverable', 'timeline'],
            'meeting': ['meeting', 'call', 'conference', 'agenda', 'schedule', 'appointment'],
            'event': ['event', 'soiree', 'gala', 'reception', 'gathering', 'celebration', 'venue', 'rsvp'],
            'budget': ['budget', 'cost', 'price', 'money', 'financial', 'expense', 'funding'],
            'technical': ['technical', 'code', 'system', 'api', 'database', 'software', 'development'],
            'hr': ['employee', 'hiring', 'interview', 'benefits', 'salary', 'performance'],
            'sales': ['sales', 'client', 'customer', 'proposal', 'contract', 'revenue'],
            'marketing': ['marketing', 'campaign', 'advertising', 'promotion', 'brand'],
            'legal': ['legal', 'contract', 'agreement', 'compliance', 'law', 'regulation'],
            'operations': ['operations', 'process', 'workflow', 'procedure', 'policy'],
            'research': ['research', 'study', 'analysis', 'data', 'findings', 'report'],
            'foundation': ['foundation', 'nonprofit', 'charity', 'donation', 'philanthropic', 'grant']
        }

        # Sentiment indicators
        self.sentiment_indicators = {
            'positive': ['great', 'excellent', 'good', 'amazing', 'wonderful', 'fantastic', 'awesome', 'perfect'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating', 'problem', 'issue'],
            'neutral': ['okay', 'fine', 'acceptable', 'standard', 'normal', 'regular']
        }

        # Action item patterns
        self.action_patterns = [
            r'(?:action item|todo|task|follow up|next step)\s*:?\s*([^.!?]*)',
            r'(?:assign|delegate|responsible|owner)\s+([^.!?]*)',
            r'(?:track|monitor|check|verify|confirm)\s+([^.!?]*)',
            r'(?:need to|should|must|will|going to)\s+([^.!?]*)'
        ]

    def enrich_email(self, email_data: Dict) -> Dict:
        """
        Enrich email data with comprehensive content analysis.
        
        Args:
            email_data: Dictionary containing email information
            
        Returns:
            Enriched email data with additional metadata
        """
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        combined_text = f"{subject} {body}"
        
        logger.info(f"🔬 EmailContentEnricher.enrich_email() called - subject: {subject[:50]}...")
        logger.info(f"🔬 Using IMPROVED entity extraction patterns (v2.0)")
        
        # Perform comprehensive analysis
        analysis = self._analyze_content(combined_text, subject)
        
        # Create enriched metadata (ensure JSON-serializable)
        enriched_data = email_data.copy()
        enriched_data.update({
            'entities': self._serialize_entities(analysis.entities),
            'topics': analysis.topics,
            'sentiment': analysis.sentiment,
            'key_phrases': analysis.key_phrases,
            'semantic_tags': analysis.semantic_tags,
            'content_type': analysis.content_type,
            'action_items': analysis.action_items,
            'relationships': analysis.relationships,
            
            # Derived fields for search optimization
            'searchable_text': self._create_searchable_text(email_data, analysis),
            'content_hash': self._generate_content_hash(combined_text),
            'enrichment_timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"✅ Enriched email with {len(analysis.entities)} entity types, {len(analysis.topics)} topics")
        logger.info(f"   Entity types found: {list(analysis.entities.keys())}")
        logger.info(f"   Topics: {analysis.topics}")
        logger.info(f"   Content type: {analysis.content_type}")
        logger.info(f"   Semantic tags: {analysis.semantic_tags}")
        return enriched_data

    def _analyze_content(self, text: str, subject: str) -> ContentAnalysis:
        """Perform comprehensive content analysis"""
        # Extract entities
        entities = self._extract_entities(text)
        
        # Classify topics
        topics = self._classify_topics(text)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(text)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(text)
        
        # Generate semantic tags
        semantic_tags = self._generate_semantic_tags(text, subject)
        
        # Classify content type
        content_type = self._classify_content_type(text, subject)
        
        # Extract action items
        action_items = self._extract_action_items(text)
        
        # Extract relationships
        relationships = self._extract_relationships(text, entities)
        
        return ContentAnalysis(
            entities=entities,
            topics=topics,
            sentiment=sentiment,
            key_phrases=key_phrases,
            semantic_tags=semantic_tags,
            content_type=content_type,
            action_items=action_items,
            relationships=relationships
        )

    def _extract_entities(self, text: str) -> Dict[str, List[EntityInfo]]:
        """Extract named entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            entity_list = []
            
            for match in matches:
                value = match.group(0)
                context = self._get_entity_context(text, match.start(), match.end())
                
                entity_info = EntityInfo(
                    type=entity_type,
                    value=value,
                    confidence=self._calculate_entity_confidence(entity_type, value, context),
                    context=context
                )
                entity_list.append(entity_info)
            
            if entity_list:
                entities[entity_type] = entity_list
        
        # Extract people (simple pattern-based approach)
        people = self._extract_people(text)
        if people:
            entities['people'] = people
        
        # Extract organizations
        organizations = self._extract_organizations(text)
        if organizations:
            entities['organizations'] = organizations
        
        # Extract locations
        locations = self._extract_locations(text)
        if locations:
            entities['locations'] = locations
        
        return entities

    def _extract_people(self, text: str) -> List[EntityInfo]:
        """Extract people names using pattern matching"""
        people = []
        seen_names = set()
        
        # Common name patterns (first name + last name)
        name_patterns = [
            r'\b[A-Z][a-z]{2,15} [A-Z][a-z]{2,15}\b',  # John Smith (2-15 chars per name)
            r'\b[A-Z]\. [A-Z][a-z]{2,15}\b',           # J. Smith
            r'\b[A-Z][a-z]{2,15} [A-Z]\. [A-Z][a-z]{2,15}\b'  # John A. Smith
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(0).strip()
                
                # Skip if already found
                if name in seen_names:
                    continue
                
                context = self._get_entity_context(text, match.start(), match.end())
                
                # Filter out common false positives
                if not self._is_likely_person(name, context):
                    continue
                
                seen_names.add(name)
                people.append(EntityInfo(
                    type='person',
                    value=name,
                    confidence=0.7,
                    context=context
                ))
        
        return people

    def _extract_organizations(self, text: str) -> List[EntityInfo]:
        """Extract organization names"""
        organizations = []
        seen_orgs = set()
        
        # Common organization patterns - more specific
        org_patterns = [
            # Explicit legal entities - Match ONLY the org name, not preceding words
            r'\b[A-Z][a-z]{2,20}(?: [A-Z][a-z]{2,20}){0,3} (?:Inc\.?|Corp\.?|LLC|Ltd\.?|Corporation|Company|Co\.)\b',
            # Known organization keywords - Match ONLY the org name
            r'\b[A-Z][a-z]{2,20}(?: [A-Z][a-z]{2,20}){0,2} (?:Foundation|Institute|University|College|Hospital|Bank|Group|Partners|Associates)\b',
            # ALL CAPS organizations (3+ letters)
            r'\b[A-Z]{3,10}\b(?! [a-z])',  # IBM, NASA, WHO (not followed by lowercase)
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                org_name = match.group(0).strip()
                
                # Remove leading articles/words like "We Are" from org name
                org_name = re.sub(r'^(?:We |The |A |An |Are )', '', org_name, flags=re.IGNORECASE).strip()
                
                # Skip if already found
                if org_name in seen_orgs:
                    continue
                
                # Skip single words that are too short
                if ' ' not in org_name and len(org_name) < 4:
                    continue
                
                # Skip common false positives
                if self._is_false_org(org_name):
                    continue
                
                context = self._get_entity_context(text, match.start(), match.end())
                
                seen_orgs.add(org_name)
                organizations.append(EntityInfo(
                    type='organization',
                    value=org_name,
                    confidence=0.8,
                    context=context
                ))
        
        return organizations

    def _is_false_org(self, org_name: str) -> bool:
        """Check if organization name is likely a false positive"""
        org_lower = org_name.lower()
        
        # Common false positives
        false_orgs = [
            'dear', 'sincerely', 'regards', 'thanks', 'best', 'hello',
            'important', 'urgent', 'action', 'required', 'information',
            'confirmation', 'reservation', 'ticket', 'final', 'exclusive'
        ]
        
        # Check if it's a common false positive
        for false_org in false_orgs:
            if false_org in org_lower:
                return True
        
        return False

    def _extract_locations(self, text: str) -> List[EntityInfo]:
        """Extract location names"""
        locations = []
        seen_locations = set()
        
        # Common city/state patterns - more specific
        location_patterns = [
            # City, State abbreviation
            r'\b[A-Z][a-z]{2,20},\s+[A-Z]{2}\b',  # New York, NY (requires comma)
            # Full addresses with street numbers
            r'\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3},\s+[A-Z][a-z]+,\s+[A-Z]{2}\s+\d{5}\b',
            # Major US cities (only well-known ones)
            r'\b(?:New York|Los Angeles|Chicago|Houston|Philadelphia|San Diego|San Francisco|Boston|Seattle|Miami|Washington DC)\b'
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                location = match.group(0).strip()
                
                # Skip if already found
                if location in seen_locations:
                    continue
                
                context = self._get_entity_context(text, match.start(), match.end())
                
                seen_locations.add(location)
                locations.append(EntityInfo(
                    type='location',
                    value=location,
                    confidence=0.7,
                    context=context
                ))
        
        return locations

    def _is_likely_person(self, name: str, context: str) -> bool:
        """Determine if a name is likely a person (not a company/product)"""
        name_lower = name.lower()
        
        # Common false positive indicators - locations, companies, generic terms
        false_positive_patterns = [
            # Locations
            r'new york|los angeles|san francisco|new jersey|san diego',
            r'united states|united kingdom|virgin hotel|virgin islands',
            # Company indicators
            r'hotel|resort|bank|corporation|company|foundation',
            # Generic terms that might match pattern
            r'final confirmation|table reservations?|ticket information',
            r'family office|dear (?:sir|madam|team|all)',
            r'action required|important notice|key (?:points?|topics?)',
            r'meeting notes?|discussion points?',
        ]
        
        # Check if name matches false positive patterns
        for pattern in false_positive_patterns:
            if re.search(pattern, name_lower):
                return False
        
        # Names that are too short or generic
        if len(name) < 6 or name.count(' ') > 2:  # Must be 6+ chars, max 2 spaces
            return False
        
        # Check if contains common title words or street indicators (likely not a person)
        non_person_words = [
            'mr', 'mrs', 'ms', 'dr', 'prof', 'rev', 'office', 'team', 'group',
            'street', 'avenue', 'road', 'boulevard', 'lane', 'drive', 'court', 'place'
        ]
        name_words = name_lower.split()
        if any(word in non_person_words for word in name_words):
            return False
        
        # Check context for person indicators (stronger signal)
        person_indicators = [
            'dear', 'hi', 'hello', 'from', 'to:', 'cc:', 
            'said', 'told', 'asked', 'replied', 'wrote', 'emailed', 'called',
            'sincerely', 'regards', 'thanks', 'contact'
        ]
        context_lower = context.lower()
        if any(indicator in context_lower for indicator in person_indicators):
            return True
        
        # Default to false to avoid false positives (conservative approach)
        return False

    def _get_entity_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Get context around an entity"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].strip()

    def _calculate_entity_confidence(self, entity_type: str, value: str, context: str) -> float:
        """Calculate confidence score for entity extraction"""
        base_confidence = {
            'email': 0.95,
            'phone': 0.9,
            'url': 0.95,
            'money': 0.9,
            'percentage': 0.95,
            'date': 0.8,
            'time': 0.85,
            'zip_code': 0.9,
            'ssn': 0.95,
            'person': 0.7,
            'organization': 0.8,
            'location': 0.7
        }.get(entity_type, 0.5)
        
        # Adjust based on context
        if len(context) > 100:  # More context = higher confidence
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence

    def _classify_topics(self, text: str) -> List[str]:
        """Classify topics based on keyword matching"""
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in self.topic_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            # If we have enough matches, include this topic
            if matches >= 2 or (matches == 1 and len(keywords) <= 3):
                topics.append(topic)
        
        return topics

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text"""
        text_lower = text.lower()
        
        # Count sentiment indicators
        positive_count = sum(1 for word in self.sentiment_indicators['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_indicators['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.sentiment_indicators['neutral'] if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            return {'overall': 0.0, 'confidence': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        # Calculate sentiment scores
        positive_score = positive_count / total_sentiment_words
        negative_score = negative_count / total_sentiment_words
        neutral_score = neutral_count / total_sentiment_words
        
        # Overall sentiment (-1 to 1)
        overall_sentiment = positive_score - negative_score
        
        # Confidence based on number of sentiment words
        confidence = min(1.0, total_sentiment_words / 5.0)
        
        return {
            'overall': overall_sentiment,
            'confidence': confidence,
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction based on patterns
        key_phrases = []
        
        # Extract noun phrases (simplified)
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter and rank phrases
        for phrase in noun_phrases:
            if len(phrase.split()) >= 2 and len(phrase) > 5:  # Multi-word phrases
                key_phrases.append(phrase)
        
        # Remove duplicates and limit
        key_phrases = list(set(key_phrases))[:10]
        
        return key_phrases

    def _generate_semantic_tags(self, text: str, subject: str) -> List[str]:
        """Generate semantic tags for better search"""
        tags = []
        text_lower = text.lower()
        subject_lower = subject.lower()
        
        # Content type tags
        if '?' in text:
            tags.append('question')
        if any(word in text_lower for word in ['deadline', 'due', 'by']):
            tags.append('deadline')
        if any(word in text_lower for word in ['meeting', 'call', 'schedule']):
            tags.append('meeting')
        if any(word in text_lower for word in ['urgent', 'asap', 'immediately']):
            tags.append('urgent')
        if any(word in text_lower for word in ['please', 'request', 'ask']):
            tags.append('request')
        if any(word in text_lower for word in ['decision', 'agree', 'decided']):
            tags.append('decision')
        if any(word in text_lower for word in ['action', 'todo', 'task']):
            tags.append('action_item')
        
        # Priority indicators
        if any(word in subject_lower for word in ['urgent', 'asap', 'important', 'critical']):
            tags.append('high_priority')
        
        # Entity type tags
        if '@' in text:
            tags.append('contact_info')
        if '$' in text:
            tags.append('financial')
        if '%' in text:
            tags.append('percentage')
        if 'http' in text_lower:
            tags.append('link')
        
        # Communication type
        if any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            tags.append('gratitude')
        if any(word in text_lower for word in ['sorry', 'apologize', 'regret']):
            tags.append('apology')
        if any(word in text_lower for word in ['congratulations', 'congrats', 'celebrate']):
            tags.append('celebration')
        
        return tags

    def _classify_content_type(self, text: str, subject: str) -> str:
        """Classify the type of email content"""
        text_lower = text.lower()
        subject_lower = subject.lower()
        
        # Meeting-related
        if any(word in text_lower for word in ['meeting', 'call', 'conference', 'schedule']):
            return 'meeting'
        
        # Question/Inquiry
        if '?' in text or any(word in text_lower for word in ['question', 'inquiry', 'ask']):
            return 'inquiry'
        
        # Request
        if any(word in text_lower for word in ['please', 'request', 'ask', 'need']):
            return 'request'
        
        # Update/Status
        if any(word in text_lower for word in ['update', 'status', 'progress', 'report']):
            return 'update'
        
        # Decision/Agreement
        if any(word in text_lower for word in ['decide', 'agree', 'approve', 'decision']):
            return 'decision'
        
        # Deadline/Urgent
        if any(word in text_lower for word in ['deadline', 'urgent', 'asap', 'due']):
            return 'deadline'
        
        # Thank you
        if any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'gratitude'
        
        # Default
        return 'general'

    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text"""
        action_items = []
        
        for pattern in self.action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                action = match.group(1).strip()
                if action and len(action) > 5:  # Meaningful action items
                    action_items.append(action)
        
        return list(set(action_items))  # Remove duplicates

    def _extract_relationships(self, text: str, entities: Dict[str, List[EntityInfo]]) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []
        
        # Simple relationship extraction based on proximity
        people = entities.get('people', [])
        organizations = entities.get('organizations', [])
        
        # Person-Organization relationships
        for person in people:
            for org in organizations:
                # Check if they appear in the same context
                if abs(person.context.find(person.value) - org.context.find(org.value)) < 100:
                    relationships.append({
                        'type': 'person_organization',
                        'source': person.value,
                        'target': org.value,
                        'confidence': 0.6
                    })
        
        return relationships

    def _create_searchable_text(self, email_data: Dict, analysis: ContentAnalysis) -> str:
        """Create enhanced searchable text combining original content with enriched data"""
        searchable_parts = []
        
        # Add original subject and body
        if email_data.get('subject'):
            searchable_parts.append(email_data['subject'])
        if email_data.get('body'):
            searchable_parts.append(email_data['body'])
        
        # Add key phrases
        if analysis.key_phrases:
            searchable_parts.append(' '.join(analysis.key_phrases))
        
        # Add topics as searchable terms
        if analysis.topics:
            searchable_parts.append(' '.join(analysis.topics))
        
        # Add entity values
        for entity_type, entities in analysis.entities.items():
            for entity in entities:
                searchable_parts.append(entity.value)
        
        # Add semantic tags
        if analysis.semantic_tags:
            searchable_parts.append(' '.join(analysis.semantic_tags))
        
        return ' '.join(searchable_parts)

    def _serialize_entities(self, entities: Dict[str, List[EntityInfo]]) -> Dict[str, List[Dict]]:
        """Convert EntityInfo objects to JSON-serializable dicts"""
        serialized = {}
        for entity_type, entity_list in entities.items():
            serialized[entity_type] = [
                {
                    'type': entity.type,
                    'value': entity.value,
                    'confidence': entity.confidence,
                    'context': entity.context[:100]  # Limit context length for storage
                }
                for entity in entity_list
            ]
        return serialized

    def _generate_content_hash(self, text: str) -> str:
        """Generate a hash for content deduplication"""
        return hashlib.md5(text.encode()).hexdigest()
