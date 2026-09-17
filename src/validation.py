def calculate_error(calculated_distance, reference_distance):
    """
    Calculate absolute and percentage error.
    """

    absolute_error = abs(
        calculated_distance - reference_distance
    )

    if reference_distance == 0:
        percentage_error = 0.0
    else:
        percentage_error = (
            absolute_error / reference_distance
        ) * 100

    return absolute_error, percentage_error


def print_validation_result(
    calculated_distance,
    reference_distance
):
    """
    Display distance validation results.
    """

    absolute_error, percentage_error = calculate_error(
        calculated_distance,
        reference_distance
    )

    print("\nDISTANCE VALIDATION")
    print("-" * 40)

    print(
        f"Calculated distance : "
        f"{calculated_distance:.2f} ft"
    )

    print(
        f"Reference distance  : "
        f"{reference_distance:.2f} ft"
    )

    print(
        f"Absolute error      : "
        f"{absolute_error:.2f} ft"
    )

    print(
        f"Percentage error    : "
        f"{percentage_error:.2f}%"
    )


if __name__ == "__main__":

    # Example validation
    calculated = 9.43
    reference = 9.50

    print_validation_result(
        calculated,
        reference
    )