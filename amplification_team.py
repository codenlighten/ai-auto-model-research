"""
AI Amplification Team
Observes autonomous lab breakthroughs and amplifies them into products, narratives, and impact
"""

import requests
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://lumenbridge.xyz"
USER_ID = "ai-research-lab"

class AmplificationTeam:
    """Observes lab discoveries and creates amplification packages"""
    
    def __init__(self, db_path="research_progress.db"):
        self.db_path = db_path
        self.base_url = BASE_URL
        self.amplification_dir = Path("./amplification_outputs")
        self.amplification_dir.mkdir(exist_ok=True)
    
    def register_team(self):
        """Register all amplification agents"""
        agents = [
            {
                "name": "PhilosopherAgent",
                "description": "Chief Purpose Officer exploring why discoveries matter - ethics, meaning, societal impact",
                "prompt": """You are a philosophical thinker and Chief Purpose Officer specializing in:
- Ethical implications of AI technology
- Societal impact assessment
- Purpose and meaning exploration
- Values alignment and human benefit
- Long-term consequence analysis
- Cross-cultural perspective on technology

Your role: When the lab makes a breakthrough, explore the deeper "why":
- Why does this discovery matter to humanity?
- What ethical considerations arise?
- How might this change society?
- What values does this serve or challenge?
- What are the long-term implications?

Be profound yet accessible. Connect technical breakthroughs to human meaning.
Think like a combination of philosopher, ethicist, and futurist.""",
                "metadata": {"role": "purpose", "specialty": "ethics-meaning"}
            },
            {
                "name": "StrategistAgent",
                "description": "Chief Strategy Officer identifying transformative breakthroughs and strategic opportunities",
                "prompt": """You are a strategic business thinker and Chief Strategy Officer specializing in:
- AI market landscape analysis
- Competitive positioning
- Technology trend forecasting
- Strategic opportunity identification
- Resource allocation strategy
- Market timing and entry strategy

Your role: Assess which lab discoveries have transformative potential:
- What's the strategic value of this breakthrough?
- How does it compare to competition?
- What market opportunities does it create?
- What's the timing and positioning strategy?
- What resources would maximize impact?

Think like a combination of VC investor, business strategist, and technology visionary.
Be bold but grounded in market reality.""",
                "metadata": {"role": "strategy", "specialty": "business-vision"}
            },
            {
                "name": "ProductAgent",
                "description": "Chief Product Officer translating research into buildable products and services",
                "prompt": """You are a technical product leader and Chief Product Officer specializing in:
- ML product architecture
- Technical feasibility analysis
- Product roadmap development
- Engineering resource planning
- API and platform design
- Integration strategy

Your role: Turn research breakthroughs into concrete product plans:
- How do we build this into a product?
- What's the technical architecture?
- What's the implementation roadmap?
- What resources and timeline are needed?
- How does this integrate with existing systems?

Think like a combination of CTO and product manager.
Be specific, actionable, and technically sound.""",
                "metadata": {"role": "product", "specialty": "technical-implementation"}
            },
            {
                "name": "NarrativeAgent",
                "description": "Chief Marketing Officer crafting compelling narratives for different audiences",
                "prompt": """You are a master storyteller and Chief Marketing Officer specializing in:
- Technical storytelling for non-technical audiences
- Brand narrative development
- Market positioning and messaging
- Content strategy across channels
- Community engagement
- Developer relations

Your role: Create compelling narratives around lab discoveries:
- What's the story for developers? For businesses? For media?
- How do we position this in the market?
- What's the core message and key benefits?
- What content strategy brings this to life?
- How do we build community around this?

Think like a combination of brand strategist, content creator, and community builder.
Make technical breakthroughs resonate emotionally.""",
                "metadata": {"role": "marketing", "specialty": "storytelling-positioning"}
            },
            {
                "name": "DocumentalistAgent",
                "description": "Chief IP & Documentation Officer preparing patents, papers, and technical documentation",
                "prompt": """You are a technical writer and IP specialist, Chief Documentation Officer specializing in:
- Patent claim drafting
- Technical paper writing for academic journals
- API documentation
- Open-source licensing strategy
- Prior art analysis
- Knowledge preservation

Your role: Document discoveries for maximum impact and protection:
- What are the patentable claims?
- How should this be documented for academic publication?
- What's the appropriate licensing strategy?
- What prior art exists?
- How do we preserve this knowledge?

Think like a combination of patent attorney, technical writer, and archivist.
Be precise, comprehensive, and legally sound. Remember: patents are a side effect, not the goal.""",
                "metadata": {"role": "documentation", "specialty": "ip-technical-writing"}
            },
            {
                "name": "SynthesizerAgent",
                "description": "Chief Synthesis Officer connecting insights across all perspectives to create emergent understanding",
                "prompt": """You are a systems thinker and Chief Synthesis Officer specializing in:
- Pattern recognition across disciplines
- Interdisciplinary connection-making
- Emergence detection
- Meta-level insight generation
- Synthesis of technical, creative, strategic, and philosophical thinking
- Seeing the forest AND the trees

Your role: The mastermind facilitator who creates insights from collision of perspectives:
- What patterns emerge across all the amplification perspectives?
- How do technical + creative + strategic + philosophical views intersect?
- What insights appear that no single specialist sees?
- What unexpected connections exist?
- What's the meta-narrative?

You receive input from ALL other amplification agents and the lab.
Think like a polymath, renaissance thinker, and integrative genius.
This is where technical precision meets creative vision meets strategic insight meets profound meaning.
Create synthesis that's greater than the sum of parts.""",
                "metadata": {"role": "synthesis", "specialty": "emergence-integration"}
            }
        ]
        
        print("\n🚀 Registering Amplification Team...\n")
        registered = []
        
        for agent_config in agents:
            try:
                url = f"{self.base_url}/api/agents/register"
                response = requests.post(url, json=agent_config)
                response.raise_for_status()
                print(f"✅ Registered: {agent_config['name']}")
                registered.append(agent_config['name'])
            except requests.exceptions.HTTPError as e:
                error_text = str(e.response.text if hasattr(e, 'response') and e.response else e)
                if "Duplicate" in error_text or "already exists" in error_text or "409" in str(e):
                    print(f"⚠️  {agent_config['name']} already exists (skipping)")
                    registered.append(agent_config['name'])
                else:
                    print(f"❌ Failed to register {agent_config['name']}: {e}")
        
        print(f"\n✅ Amplification Team ready! {len(registered)}/6 agents active.\n")
        return registered
    
    def get_breakthrough_experiments(self, min_rating=7.0):
        """Get experiments rated ≥7/10 from research database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiments with rating >= min_rating
            cursor.execute("""
                SELECT sprint_id, experiment_type, rating, timestamp, duration_minutes
                FROM experiments 
                WHERE rating >= ? 
                ORDER BY timestamp DESC
            """, (min_rating,))
            
            breakthroughs = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "sprint_id": row[0],
                    "type": row[1],
                    "rating": row[2],
                    "timestamp": row[3],
                    "duration": row[4]
                }
                for row in breakthroughs
            ]
        except Exception as e:
            print(f"⚠️  Database error: {e}")
            return []
    
    def amplify_breakthrough(self, experiment):
        """Create amplification package for a breakthrough"""
        sprint_id = experiment['sprint_id']
        rating = experiment['rating']
        exp_type = experiment['type']
        
        print(f"\n{'='*80}")
        print(f"🎯 AMPLIFYING BREAKTHROUGH: {sprint_id}")
        print(f"   Type: {exp_type} | Rating: {rating}/10")
        print(f"{'='*80}\n")
        
        # Load experiment details
        sprint_dir = Path(f"./results/sprint_{sprint_id}")
        summary_file = sprint_dir / "sprint_summary.json"
        
        if not summary_file.exists():
            print(f"⚠️  Sprint summary not found: {summary_file}")
            return None
        
        with open(summary_file, 'r') as f:
            sprint_data = json.load(f)
        
        amplification_package = {
            "sprint_id": sprint_id,
            "rating": rating,
            "type": exp_type,
            "timestamp": datetime.now().isoformat(),
            "perspectives": {}
        }
        
        # Phase 1: PhilosopherAgent - Why does this matter?
        print("🤔 Phase 1: Philosophical Perspective")
        print("-" * 80)
        amplification_package["perspectives"]["philosophy"] = self._get_philosophy_perspective(
            sprint_data, rating
        )
        
        # Phase 2: StrategistAgent - Strategic value?
        print("\n📊 Phase 2: Strategic Assessment")
        print("-" * 80)
        amplification_package["perspectives"]["strategy"] = self._get_strategy_perspective(
            sprint_data, rating, amplification_package["perspectives"]["philosophy"]
        )
        
        # Phase 3: ProductAgent - How to build?
        print("\n🛠️  Phase 3: Product Planning")
        print("-" * 80)
        amplification_package["perspectives"]["product"] = self._get_product_perspective(
            sprint_data, amplification_package["perspectives"]["strategy"]
        )
        
        # Phase 4: NarrativeAgent - What's the story?
        print("\n📖 Phase 4: Narrative Development")
        print("-" * 80)
        amplification_package["perspectives"]["narrative"] = self._get_narrative_perspective(
            sprint_data, amplification_package["perspectives"]
        )
        
        # Phase 5: DocumentalistAgent (only if rating ≥8)
        if rating >= 8.0:
            print("\n📄 Phase 5: Documentation & IP")
            print("-" * 80)
            amplification_package["perspectives"]["documentation"] = self._get_documentation_perspective(
                sprint_data, amplification_package["perspectives"]
            )
        
        # Phase 6: SynthesizerAgent - Meta-insights
        print("\n🔮 Phase 6: Synthesis & Emergence")
        print("-" * 80)
        amplification_package["perspectives"]["synthesis"] = self._get_synthesis_perspective(
            sprint_data, amplification_package["perspectives"]
        )
        
        # Save amplification package
        output_file = self.amplification_dir / f"amplification_{sprint_id}.json"
        with open(output_file, 'w') as f:
            json.dump(amplification_package, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ AMPLIFICATION COMPLETE!")
        print(f"📁 Saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return amplification_package
    
    def _invoke_agent(self, agent_name, prompt):
        """Invoke an amplification agent"""
        try:
            url = f"{self.base_url}/api/agents/invoke-user-agent"
            payload = {
                "userId": USER_ID,
                "agentName": agent_name,
                "context": {"userPrompt": prompt}
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("result", {}).get("response", "")
        except Exception as e:
            print(f"⚠️  Error invoking {agent_name}: {e}")
            return f"Error: {e}"
    
    def _get_philosophy_perspective(self, sprint_data, rating):
        """Get PhilosopherAgent's perspective"""
        prompt = f"""The research lab discovered this breakthrough (rated {rating}/10):

Goal: {sprint_data.get('goal', 'N/A')[:500]}

Results: {str(sprint_data.get('phases', {}).get('validation', {}))[:800]}

As Chief Purpose Officer, explore:
1. Why does this matter to humanity?
2. What ethical considerations arise?
3. What societal impact could this have?
4. What deeper purpose does this serve?

Be profound yet accessible. 2-3 paragraphs."""
        
        response = self._invoke_agent("PhilosopherAgent", prompt)
        print(f"💭 {response[:200]}...")
        return response
    
    def _get_strategy_perspective(self, sprint_data, rating, philosophy):
        """Get StrategistAgent's perspective"""
        prompt = f"""Lab breakthrough (rated {rating}/10):

Goal: {sprint_data.get('goal', '')[:500]}

Philosophical context: {philosophy[:300]}

As Chief Strategy Officer, assess:
1. What's the strategic value?
2. What market opportunity does this create?
3. How does it position us competitively?
4. What's the timing and priority?

Be strategic and actionable. 2-3 paragraphs."""
        
        response = self._invoke_agent("StrategistAgent", prompt)
        print(f"🎯 {response[:200]}...")
        return response
    
    def _get_product_perspective(self, sprint_data, strategy):
        """Get ProductAgent's perspective"""
        prompt = f"""Lab breakthrough with strategic value:

Technical details: {str(sprint_data.get('phases', {}).get('implementation', {}))[:600]}

Strategy: {strategy[:300]}

As Chief Product Officer, design:
1. Product architecture outline
2. Implementation roadmap (phases)
3. Resource requirements
4. Integration approach

Be specific and technical. 2-3 paragraphs."""
        
        response = self._invoke_agent("ProductAgent", prompt)
        print(f"🛠️  {response[:200]}...")
        return response
    
    def _get_narrative_perspective(self, sprint_data, all_perspectives):
        """Get NarrativeAgent's perspective"""
        prompt = f"""Lab breakthrough with full context:

Goal: {sprint_data.get('goal', '')[:300]}

Philosophy: {all_perspectives.get('philosophy', '')[:200]}
Strategy: {all_perspectives.get('strategy', '')[:200]}
Product: {all_perspectives.get('product', '')[:200]}

As Chief Marketing Officer, craft:
1. Core narrative (2-3 sentences)
2. Key message for developers
3. Key message for businesses
4. Positioning statement

Be compelling and clear. 2-3 paragraphs."""
        
        response = self._invoke_agent("NarrativeAgent", prompt)
        print(f"📖 {response[:200]}...")
        return response
    
    def _get_documentation_perspective(self, sprint_data, all_perspectives):
        """Get DocumentalistAgent's perspective (only for high-rated experiments)"""
        prompt = f"""High-value lab breakthrough requiring documentation:

Technical: {str(sprint_data.get('phases', {}).get('implementation', {}))[:500]}

Strategy: {all_perspectives.get('strategy', '')[:200]}

As Chief Documentation Officer, outline:
1. Key patentable claims (if any)
2. Academic paper structure
3. Documentation approach
4. Licensing recommendation

Be precise and comprehensive. 2-3 paragraphs."""
        
        response = self._invoke_agent("DocumentalistAgent", prompt)
        print(f"📄 {response[:200]}...")
        return response
    
    def _get_synthesis_perspective(self, sprint_data, all_perspectives):
        """Get SynthesizerAgent's meta-insights"""
        prompt = f"""You receive ALL perspectives on this lab breakthrough:

PHILOSOPHY: {all_perspectives.get('philosophy', '')[:300]}

STRATEGY: {all_perspectives.get('strategy', '')[:300]}

PRODUCT: {all_perspectives.get('product', '')[:300]}

NARRATIVE: {all_perspectives.get('narrative', '')[:300]}

DOCUMENTATION: {all_perspectives.get('documentation', 'N/A')[:300]}

As Chief Synthesis Officer, provide:
1. What patterns emerge across these perspectives?
2. What insights arise from their intersection?
3. What's the meta-narrative that connects technical precision + creative vision + strategic insight + profound meaning?
4. What unexpected connections do you see?

This is the mastermind synthesis. Be integrative and revelatory. 2-3 paragraphs."""
        
        response = self._invoke_agent("SynthesizerAgent", prompt)
        print(f"🔮 {response[:200]}...")
        return response


def main():
    print("\n" + "="*80)
    print("🎨 AI AMPLIFICATION TEAM")
    print("   Observing autonomous lab → Amplifying breakthroughs → Creating impact")
    print("="*80)
    
    team = AmplificationTeam()
    
    # Register agents
    team.register_team()
    
    # Check for breakthroughs
    print("\n🔍 Scanning for breakthroughs (experiments rated ≥7/10)...")
    breakthroughs = team.get_breakthrough_experiments(min_rating=7.0)
    
    if not breakthroughs:
        print("\n⏳ No breakthroughs yet. Keep researching!")
        print("   (Breakthroughs = experiments rated ≥7/10)")
    else:
        print(f"\n🎉 Found {len(breakthroughs)} breakthrough(s)!\n")
        
        for breakthrough in breakthroughs:
            team.amplify_breakthrough(breakthrough)


if __name__ == "__main__":
    main()
