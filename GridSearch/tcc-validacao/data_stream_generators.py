from skmultiflow.data import (
    ConceptDriftStream,
    SEAGenerator,
    AGRAWALGenerator,
    FileStream,
)


def generate_sea_with_drift(size: int, width: int = 1, random_state: int = 1):
    drift1 = ConceptDriftStream(
        SEAGenerator(random_state=random_state),
        SEAGenerator(random_state=random_state, classification_function=1),
        position=size * 0.25,
        width=width,
        random_state=random_state,
    )
    drift2 = ConceptDriftStream(
        drift1,
        SEAGenerator(random_state=random_state, classification_function=2),
        position=size * 0.5,
        width=width,
        random_state=random_state,
    )
    return ConceptDriftStream(
        drift2,
        SEAGenerator(random_state=random_state, classification_function=3),
        position=size * 0.75,
        width=width,
        random_state=random_state,
    )


def generate_agr_with_drift(size: int, width: int = 1, random_state: int = 1):
    drift1 = ConceptDriftStream(
        AGRAWALGenerator(random_state=random_state),
        AGRAWALGenerator(random_state=random_state, classification_function=1),
        position=size * 0.25,
        width=width,
        random_state=random_state,
    )
    drift2 = ConceptDriftStream(
        drift1,
        AGRAWALGenerator(random_state=random_state, classification_function=2),
        position=size * 0.5,
        width=width,
        random_state=random_state,
    )
    return ConceptDriftStream(
        drift2,
        AGRAWALGenerator(random_state=random_state, classification_function=3),
        position=size * 0.75,
        width=width,
        random_state=random_state,
    )


def get_dataset(dataset, size):
    if dataset == "agr_a":
        return generate_agr_with_drift(size=size)
    if dataset == "agr_g":
        return generate_agr_with_drift(size=size, width=int(size * 0.02))
    if dataset == "sea_a":
        return generate_sea_with_drift(size=size)
    if dataset == "sea_g":
        return generate_sea_with_drift(size=size, width=int(size * 0.02))
    return FileStream(f"datasets/{dataset}.csv")
