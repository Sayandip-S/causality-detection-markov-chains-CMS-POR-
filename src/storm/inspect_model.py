from model_utils import load_prism_model


def main() -> None:
    _, _, model = load_prism_model()

    print("Model summary")
    print("-------------")
    print(f"Type: {model.model_type}")
    print(f"States: {model.nr_states}")
    print(f"Transitions: {model.nr_transitions}")
    print(f"Initial states: {list(model.initial_states)}")
    print()

    print("State information")
    print("-----------------")

    for state in model.states:
        labels = list(state.labels)

        print(
            f"State ID: {state.id}, "
            f"Labels: {labels}"
        )

        for action in state.actions:
            for transition in action.transitions:
                print(
                    "  -> "
                    f"state {transition.column} "
                    f"with probability {float(transition.value()):.6f}"
                )

        print()


if __name__ == "__main__":
    main()