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

        valuation = model.state_valuations.get_string(state.id)

        print(
            f"Storm state ID: {state.id}, "
            f"PRISM valuation: {valuation}, "
            f"Labels: {labels}"
        )

        for action in state.actions:
            for transition in action.transitions:
                target_id = transition.column
                probability = float(transition.value())

                target_valuation = (
                    model.state_valuations.get_string(target_id)
                )

                print(
                    f"  -> Storm state {target_id} "
                    f"({target_valuation}) "
                    f"with probability {probability:.6f}"
                )

        print()


if __name__ == "__main__":
    main()