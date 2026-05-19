from ctm_for_mapf.envs import Action, AgentSpec
from ctm_for_mapf.models import CTMRecoveryCore


def main() -> None:
    agent = AgentSpec(agent_id=0, start=(0, 0), goal=(2, 2))
    print(f"agent={agent}")
    print(f"actions={[action.value for action in Action]}")
    print(f"ctm_core={CTMRecoveryCore.__name__}")


if __name__ == "__main__":
    main()
