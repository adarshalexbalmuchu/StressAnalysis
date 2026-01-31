"""
Complete End-to-End Example Workflow

This script demonstrates the full lifecycle of a stress-test run:
1. Create run
2. Load hypotheses
3. Build graph
4. Initialize beliefs
5. Ingest signals
6. Update beliefs
7. Run simulation
8. Analyze results

Run with: poetry run python examples/complete_workflow.py
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import requests

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
SEED_FILE = Path(__file__).parent.parent / "data" / "seeds" / "pds_biometric_v1.json"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main() -> None:
    """Execute complete workflow."""
    
    print_section("PDS STRESS TEST - COMPLETE WORKFLOW")
    
    # Load seed data
    print("📁 Loading seed data...")
    with open(SEED_FILE) as f:
        seed_data = json.load(f)
    print(f"  ✓ Loaded {len(seed_data['hypotheses'])} hypotheses")
    print(f"  ✓ Loaded {len(seed_data['graph_edges'])} edges")
    
    # Step 1: Create Run
    print_section("Step 1: Create Run")
    run_response = requests.post(
        f"{BASE_URL}/runs",
        json={
            "policy_rule": seed_data["policy_rule"],
            "domain": seed_data["domain"],
            "description": seed_data["description"],
            "metadata": {
                "created_by": "example_workflow",
                "seed_file": "pds_biometric_v1.json"
            }
        }
    )
    run_response.raise_for_status()
    run = run_response.json()
    run_id = run["id"]
    
    print(f"✓ Created run: {run_id}")
    print(f"  Policy: {run['policy_rule'][:60]}...")
    print(f"  Status: {run['status']}")
    
    # Step 2: Load Hypotheses
    print_section("Step 2: Load Hypotheses")
    hypotheses_response = requests.post(
        f"{BASE_URL}/hypotheses/bulk/{run_id}",
        json=seed_data["hypotheses"]
    )
    hypotheses_response.raise_for_status()
    hypotheses = hypotheses_response.json()
    
    print(f"✓ Loaded {len(hypotheses)} hypotheses")
    print(f"  Sample: {hypotheses[0]['hid']} - {hypotheses[0]['mechanism'][:50]}...")
    
    # Step 3: Build Graph
    print_section("Step 3: Build Hypothesis Graph")
    graph_response = requests.post(
        f"{BASE_URL}/graph/{run_id}",
        json=seed_data["graph_edges"]
    )
    graph_response.raise_for_status()
    graph = graph_response.json()
    
    print(f"✓ Built graph:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['edges'])}")
    print(f"  Density: {graph['metadata']['density']:.3f}")
    print(f"  Components: {len(graph['metadata']['components'])}")
    
    # Show top central hypotheses
    centrality = graph['metadata']['centrality']
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top 5 Central Hypotheses:")
    for hid, score in top_central:
        print(f"    {hid}: {score:.3f}")
    
    # Step 4: Initialize Beliefs
    print_section("Step 4: Initialize Belief State")
    beliefs_response = requests.post(
        f"{BASE_URL}/beliefs/{run_id}/initialize",
        params={"prior_belief": 0.5}
    )
    beliefs_response.raise_for_status()
    beliefs = beliefs_response.json()
    
    print(f"✓ Initialized beliefs:")
    print(f"  Total hypotheses: {len(beliefs['beliefs'])}")
    print(f"  Prior: 0.5 (uniform)")
    print(f"  Explanation log entries: {len(beliefs['explanation_log'])}")
    
    # Step 5: Create Sample Signals
    print_section("Step 5: Ingest Evidence Signals")
    
    # Create realistic signals based on the seed data
    sample_signals = [
        {
            "signal_type": "grievance",
            "content": "Multiple complaints from elderly beneficiaries in rural Maharashtra about fingerprint authentication failures",
            "source": "District Collector's Office, Pune",
            "date_observed": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "affected_hids": ["H001", "H009"],
            "strength": 0.7,
            "metadata": {"district": "Pune", "complaints_count": 142}
        },
        {
            "signal_type": "field_report",
            "content": "Field assessment shows 35% authentication failure rate among manual laborers at 15 FPS locations",
            "source": "Civil Society Monitoring Report",
            "date_observed": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
            "affected_hids": ["H002", "H009"],
            "strength": 0.8,
            "metadata": {"sample_size": 1500, "locations": 15}
        },
        {
            "signal_type": "audit",
            "content": "State audit reveals reduced PDS leakage from 22% to 8% post-biometric implementation",
            "source": "State Comptroller and Auditor General",
            "date_observed": (datetime.now(timezone.utc) - timedelta(days=15)).isoformat(),
            "affected_hids": ["H014", "H015"],
            "strength": 0.9,
            "metadata": {"leakage_before": 0.22, "leakage_after": 0.08}
        },
        {
            "signal_type": "media_report",
            "content": "News coverage of starvation death linked to ration denial due to Aadhaar authentication failure",
            "source": "The Hindu, Front Page",
            "date_observed": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            "affected_hids": ["H001", "H009", "H018"],
            "strength": 0.6,
            "metadata": {"media_reach": "national", "public_attention": "high"}
        },
        {
            "signal_type": "court_observation",
            "content": "High Court notice to state government on PIL regarding biometric exclusion of disabled beneficiaries",
            "source": "Bombay High Court",
            "date_observed": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
            "affected_hids": ["H012", "H009"],
            "strength": 0.7,
            "metadata": {"case_number": "PIL/2026/123", "status": "notice_issued"}
        }
    ]
    
    signals_response = requests.post(
        f"{BASE_URL}/signals/{run_id}/bulk",
        json=sample_signals
    )
    signals_response.raise_for_status()
    signals = signals_response.json()
    
    print(f"✓ Created {len(signals)} evidence signals:")
    for sig in signals:
        print(f"  • {sig['signal_type']}: {sig['content'][:60]}...")
    
    # Step 6: Update Beliefs
    print_section("Step 6: Update Beliefs with Evidence")
    
    signal_ids = [sig["id"] for sig in signals]
    update_response = requests.post(
        f"{BASE_URL}/beliefs/{run_id}/update",
        json={
            "signal_ids": signal_ids,
            "update_method": "bayesian_simple",
            "notes": "Incorporating field evidence from past 30 days"
        }
    )
    update_response.raise_for_status()
    updated_beliefs = update_response.json()
    
    print(f"✓ Updated belief state:")
    
    # Show beliefs that changed significantly
    old_beliefs = beliefs['beliefs']
    new_beliefs = updated_beliefs['beliefs']
    
    changes = []
    for hid in new_beliefs:
        if hid in old_beliefs:
            change = new_beliefs[hid] - old_beliefs[hid]
            if abs(change) > 0.05:
                changes.append((hid, old_beliefs[hid], new_beliefs[hid], change))
    
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    
    print(f"\n  Significant belief updates (|Δ| > 0.05):")
    for hid, old, new, delta in changes[:10]:
        direction = "↑" if delta > 0 else "↓"
        print(f"    {hid}: {old:.3f} → {new:.3f} ({direction} {abs(delta):.3f})")
    
    # Step 7: Run Simulation
    print_section("Step 7: Run Monte Carlo Simulation")
    
    print("Running simulation (this may take 10-30 seconds)...")
    simulation_response = requests.post(
        f"{BASE_URL}/simulation/{run_id}",
        params={
            "n_trajectories": 1000,
            "max_steps": 10,
            "activation_threshold": 0.5,
            "propagation_strength": 0.3,
            "random_seed": 42
        }
    )
    simulation_response.raise_for_status()
    simulation = simulation_response.json()
    
    print("✓ Simulation complete!")
    
    # Step 8: Analyze Results
    print_section("Step 8: Analyze Simulation Results")
    
    stats = simulation["summary_statistics"]
    
    print(f"Simulation Overview:")
    print(f"  Total trajectories: {stats['total_trajectories']}")
    print(f"  Successful: {stats['completed_trajectories']} ({stats['success_rate']:.1%})")
    print(f"  Failed: {stats['total_trajectories'] - stats['completed_trajectories']} ({stats['failure_rate']:.1%})")
    print(f"  Avg trajectory length: {stats['avg_trajectory_length']:.1f} steps")
    print(f"  Std trajectory length: {stats['std_trajectory_length']:.1f} steps")
    
    print(f"\nMost Frequently Activated Hypotheses:")
    for item in stats["most_frequent_activations"][:10]:
        print(f"  {item['hid']}: {item['frequency']:.1%} of trajectories")
    
    print(f"\nMost Frequent Effects:")
    for item in stats["most_frequent_effects"][:5]:
        print(f"  • {item['effect']}")
        print(f"    (appears in {item['frequency']:.1%} of trajectories)")
    
    print(f"\nSensitivity Hotspots (high-variance hypotheses):")
    hotspots = simulation["sensitivity_hotspots"]
    for hid in hotspots[:10]:
        print(f"  • {hid}")
    
    # Sample trajectory analysis
    print(f"\nSample Trajectory (first successful):")
    sample_traj = None
    for traj in simulation["trajectories"]:
        if traj["status"] == "completed" and len(traj["steps"]) > 0:
            sample_traj = traj
            break
    
    if sample_traj:
        print(f"  Trajectory ID: {sample_traj['trajectory_id']}")
        print(f"  Steps: {len(sample_traj['steps'])}")
        print(f"  Total activated: {sample_traj['final_state']['total_activated_count']}")
        print(f"  Unique effects: {sample_traj['final_state']['unique_effects']}")
        print(f"\n  Top effects in this trajectory:")
        for effect in sample_traj['final_state']['top_effects']:
            print(f"    • {effect}")
    
    # Final Summary
    print_section("Summary")
    
    print(f"Run ID: {run_id}")
    print(f"Status: Completed")
    print(f"")
    print(f"Key Insights:")
    print(f"  1. System modeled {len(hypotheses)} mechanistic hypotheses")
    print(f"  2. Evidence signals updated {len(changes)} hypothesis beliefs significantly")
    print(f"  3. Simulation explored {stats['total_trajectories']} possible future trajectories")
    print(f"  4. Success rate: {stats['success_rate']:.1%}")
    print(f"  5. Identified {len(hotspots)} sensitivity hotspots requiring attention")
    print(f"")
    print(f"Next steps:")
    print(f"  • Review sensitivity hotspots for policy design modifications")
    print(f"  • Analyze trajectory clusters to identify distinct future scenarios")
    print(f"  • Incorporate additional signals as they become available")
    print(f"  • Re-run simulation with different parameters for sensitivity analysis")
    print(f"")
    print(f"View full results at:")
    print(f"  http://localhost:8000/api/v1/simulation/{run_id}/latest")
    print(f"")
    print("✅ Workflow complete!")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("   Please ensure the server is running:")
        print("   poetry run uvicorn app.main:app --reload")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ API Error: {e}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
