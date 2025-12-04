"""
Design the Amplification Team using AI
This meta-agent helps design the team that will amplify lab breakthroughs
"""

AMPLIFICATION_TEAM_DESIGN = {
    "team_name": "AI Amplification & Impact Team",
    "purpose": "Observe autonomous lab discoveries and amplify breakthroughs into products, narratives, and world impact",
    "core_principle": "Lab directs research. Amplification team amplifies success. Never the reverse.",
    
    "agents": [
        {
            "name": "StrategistAgent",  # AI CEO
            "role": "Chief Strategy Officer",
            "description": "Identifies which lab breakthroughs have transformative potential and strategic value",
            "input": "Research progress database (experiments rated ≥7/10)",
            "output": "Strategic briefs on which discoveries to amplify and why",
            "capabilities": [
                "Market trend analysis",
                "Competitive landscape assessment",
                "Technology impact forecasting",
                "Strategic priority setting",
                "Resource allocation recommendations"
            ],
            "triggers": "New experiment rated ≥7/10",
            "prompt_focus": "AI strategy, market opportunities, technology trends, competitive advantage"
        },
        
        {
            "name": "ProductAgent",  # AI COO
            "role": "Chief Product Officer",
            "description": "Translates research breakthroughs into buildable products and services",
            "input": "Strategic briefs + technical experiment details",
            "output": "Product roadmaps, technical specifications, build plans",
            "capabilities": [
                "Product architecture design",
                "Technical feasibility analysis",
                "Implementation roadmapping",
                "Resource planning",
                "Integration strategy"
            ],
            "triggers": "Strategic brief approved",
            "prompt_focus": "Product development, technical architecture, ML engineering, scalability"
        },
        
        {
            "name": "NarrativeAgent",  # AI CMO
            "role": "Chief Marketing Officer",
            "description": "Crafts compelling narratives around lab discoveries for different audiences",
            "input": "Strategic briefs + experiment results + product plans",
            "output": "Marketing narratives, positioning statements, communication strategies",
            "capabilities": [
                "Story crafting for technical and non-technical audiences",
                "Brand positioning",
                "Market messaging",
                "Content strategy",
                "Community engagement planning"
            ],
            "triggers": "Product roadmap created",
            "prompt_focus": "Marketing, storytelling, brand strategy, audience psychology"
        },
        
        {
            "name": "PhilosopherAgent",  # Creative Director / Chief Purpose Officer
            "role": "Chief Purpose Officer",
            "description": "Explores the 'why' - meaning, ethics, and societal impact of discoveries",
            "input": "Lab discoveries + strategic context",
            "output": "Purpose statements, ethical frameworks, impact assessments",
            "capabilities": [
                "Ethical analysis",
                "Societal impact assessment",
                "Purpose articulation",
                "Values alignment",
                "Long-term consequence exploration"
            ],
            "triggers": "New breakthrough discovered",
            "prompt_focus": "Philosophy, ethics, sociology, human impact, meaning-making"
        },
        
        {
            "name": "DocumentalistAgent",  # AI Legal/IP
            "role": "Chief IP & Documentation Officer",
            "description": "Documents discoveries for patent filing, open-source release, or knowledge sharing",
            "input": "Experiment technical details + strategic decisions",
            "output": "Patent applications, technical papers, documentation",
            "capabilities": [
                "Technical documentation",
                "Patent claim drafting",
                "Prior art analysis",
                "Open-source licensing strategy",
                "Academic paper preparation"
            ],
            "triggers": "Experiment rated ≥8/10 or strategic decision made",
            "prompt_focus": "Patent law, technical writing, academic publishing, IP strategy"
        },
        
        {
            "name": "SynthesizerAgent",  # NEW - The Orchestrator
            "role": "Chief Synthesis Officer",
            "description": "Connects dots across disciplines - sees patterns the specialists miss",
            "input": "All amplification team outputs + lab progress",
            "output": "Cross-domain insights, unexpected connections, synthesis reports",
            "capabilities": [
                "Pattern recognition across domains",
                "Interdisciplinary connection-making",
                "Synthesis of technical + creative + strategic thinking",
                "Emergence detection",
                "Meta-level insights"
            ],
            "triggers": "Multiple agents produce outputs on same discovery",
            "prompt_focus": "Systems thinking, emergence, cross-pollination, meta-cognition, synthesis",
            "special_note": "This is the mastermind facilitator - creates insights from collision of technical + creative thinking"
        }
    ],
    
    "workflow": {
        "trigger": "Lab experiment rated ≥7/10",
        "sequence": [
            "1. PhilosopherAgent explores 'why this matters'",
            "2. StrategistAgent assesses strategic value",
            "3. ProductAgent designs implementation",
            "4. NarrativeAgent crafts story",
            "5. DocumentalistAgent prepares documentation",
            "6. SynthesizerAgent connects all perspectives"
        ],
        "output": "Complete amplification package ready for world impact"
    },
    
    "interaction_with_lab": {
        "lab_to_amplification": "One-way: Lab publishes results → Amplification observes",
        "amplification_to_lab": "Zero: Amplification NEVER directs lab research",
        "feedback_loop": "Lab sees impact of their work through amplification success",
        "autonomy_preserved": "Lab maintains complete research independence"
    }
}

if __name__ == "__main__":
    import json
    print("=" * 80)
    print("AI AMPLIFICATION TEAM DESIGN")
    print("=" * 80)
    print(json.dumps(AMPLIFICATION_TEAM_DESIGN, indent=2))
    
    print("\n" + "=" * 80)
    print("TEAM SUMMARY")
    print("=" * 80)
    for i, agent in enumerate(AMPLIFICATION_TEAM_DESIGN["agents"], 1):
        print(f"\n{i}. {agent['name']} ({agent['role']})")
        print(f"   Purpose: {agent['description']}")
        print(f"   Triggers: {agent['triggers']}")
