"""
main.py — Entry point for the AI Agent Auditor

Usage:
  python main.py audit    <spec.json> [output.json]   # audit one agent spec
  python main.py eval                                  # run full eval suite
  python main.py optimize                              # run optimizer loop
"""

import asyncio
import sys


def print_usage():
    print(__doc__)


async def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "audit":
        from agent.auditor import run_auditor
        spec  = sys.argv[2] if len(sys.argv) > 2 else "data/sample_agents/customer_support.json"
        out   = sys.argv[3] if len(sys.argv) > 3 else "results/audit_output.json"
        await run_auditor(spec, out)

    elif command == "eval":
        from eval.harness import run_evals
        await run_evals(tag="manual_run")

    elif command == "optimize":
        from optimizer.optimize import run_optimizer
        await run_optimizer()

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
