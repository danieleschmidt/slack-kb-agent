#!/usr/bin/env python3
"""
Autonomous SDLC Execution Demo

This script demonstrates the quantum-inspired task planning and autonomous SDLC
execution capabilities added to the Slack KB Agent.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from slack_kb_agent.autonomous_sdlc import get_autonomous_sdlc
from slack_kb_agent.quantum_task_planner import get_quantum_planner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_quantum_task_planning():
    """Demonstrate quantum task planning capabilities."""
    logger.info("🌟 Demonstrating Quantum Task Planning")
    
    planner = get_quantum_planner()
    
    # Create some demo tasks
    def sample_task_1(task):
        logger.info(f"Executing task: {task.name}")
        return True
    
    def sample_task_2(task):
        logger.info(f"Executing task: {task.name}")
        return True
    
    def sample_task_3(task):
        logger.info(f"Executing task: {task.name}")
        return True
    
    # Create tasks with different priorities and quantum states
    task1 = planner.create_task(
        "Initialize System", 
        "Initialize core system components",
        sample_task_1,
        superposition_states=["ready", "initializing", "waiting"]
    )
    
    task2 = planner.create_task(
        "Process Data",
        "Process incoming data streams", 
        sample_task_2,
        dependencies=[task1.id],
        superposition_states=["ready", "processing", "blocked"]
    )
    
    task3 = planner.create_task(
        "Generate Report",
        "Generate system report",
        sample_task_3,
        dependencies=[task2.id],
        superposition_states=["ready", "generating", "complete"]
    )
    
    # Demonstrate quantum entanglement
    planner.entangle_tasks(task1.id, task3.id)
    
    logger.info(f"Created {len(planner.tasks)} quantum tasks")
    logger.info(f"Task 1 probability: {task1.calculate_execution_probability():.2f}")
    logger.info(f"Task 2 probability: {task2.calculate_execution_probability():.2f}")
    logger.info(f"Task 3 probability: {task3.calculate_execution_probability():.2f}")
    
    # Measure quantum states
    state1 = planner.measure_task_state(task1.id)
    state2 = planner.measure_task_state(task2.id)
    state3 = planner.measure_task_state(task3.id)
    
    logger.info(f"Measured states - Task1: {state1}, Task2: {state2}, Task3: {state3}")
    
    # Schedule and execute tasks
    success1 = planner.schedule_task(task1.id)
    success2 = planner.schedule_task(task2.id)  # Should be blocked by dependency
    success3 = planner.schedule_task(task3.id)  # Should be blocked by dependency
    
    logger.info(f"Scheduling results - Task1: {success1}, Task2: {success2}, Task3: {success3}")
    
    # Run execution loop for a short time
    logger.info("Starting quantum execution loop...")
    
    # Create a task to run the execution loop with timeout
    try:
        await asyncio.wait_for(
            planner.run_execution_loop(),
            timeout=10.0  # Run for 10 seconds
        )
    except asyncio.TimeoutError:
        logger.info("Execution loop demo completed")
    
    # Get system metrics
    metrics = planner.get_system_metrics()
    logger.info("Quantum System Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


async def demo_autonomous_sdlc():
    """Demonstrate autonomous SDLC execution."""
    logger.info("🚀 Demonstrating Autonomous SDLC Execution")
    
    # Get the autonomous SDLC engine
    sdlc = get_autonomous_sdlc()
    
    # Analyze the project
    logger.info("Analyzing project structure...")
    project_info = sdlc.analyze_project_structure()
    
    logger.info("Project Analysis Results:")
    for key, value in project_info.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Execute a subset of SDLC phases for demo
    from slack_kb_agent.autonomous_sdlc import SDLCPhase
    
    demo_phases = [
        SDLCPhase.ANALYSIS,
        SDLCPhase.TESTING,
    ]
    
    for phase in demo_phases:
        logger.info(f"Executing SDLC Phase: {phase.phase_name}")
        success = await sdlc.execute_sdlc_phase(phase)
        logger.info(f"Phase {phase.phase_name} {'✅ succeeded' if success else '❌ failed'}")
    
    # Get execution summary
    summary = sdlc.get_execution_summary()
    logger.info("SDLC Execution Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")


async def main():
    """Main demo function."""
    logger.info("🌈 Starting Autonomous SDLC and Quantum Task Planning Demo")
    logger.info("=" * 60)
    
    try:
        # Demo quantum task planning
        await demo_quantum_task_planning()
        
        logger.info("=" * 60)
        
        # Demo autonomous SDLC
        await demo_autonomous_sdlc()
        
        logger.info("=" * 60)
        logger.info("✨ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())