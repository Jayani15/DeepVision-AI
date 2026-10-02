import json
import random
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "lfw-deepfunneled"
)

GALLERY_DIR = BASE_DIR / "gallery"

SPLIT_FILE = GALLERY_DIR / "split.json"


# Number of identities we want to use
NUM_KNOWN_IDENTITIES = 50
NUM_UNKNOWN_IDENTITIES = 20

# Images per known identity
GALLERY_IMAGES_PER_PERSON = 10
TEST_IMAGES_PER_PERSON = 5

RANDOM_SEED = 42


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# Find LFW person folders
# ============================================================

def find_person_directories():

    person_directories = []

    for directory in DATASET_DIR.rglob("*"):

        if not directory.is_dir():
            continue

        images = [
            file
            for file in directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images:
            person_directories.append(
                {
                    "name": directory.name,
                    "path": directory,
                    "images": sorted(images),
                }
            )

    return person_directories


# ============================================================
# Main
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("LFW DATASET PREPARATION")
    print("=" * 60)

    print()
    print(f"Dataset directory:")
    print(DATASET_DIR)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nLFW dataset directory not found:\n"
            f"{DATASET_DIR}\n"
        )

    # --------------------------------------------------------
    # Find people
    # --------------------------------------------------------

    people = find_person_directories()

    print()
    print(
        f"Person folders found: {len(people)}"
    )

    if not people:

        print()
        print(
            "ERROR: No person folders containing "
            "images were found."
        )

        print()
        print("Expected structure:")
        print(
            "lfw-deepfunneled/"
            "Person_Name/"
            "Person_Name_0001.jpg"
        )

        raise RuntimeError(
            "No LFW person directories found."
        )

    # --------------------------------------------------------
    # Filter people based on image count
    # --------------------------------------------------------

    required_images = (
        GALLERY_IMAGES_PER_PERSON
        + TEST_IMAGES_PER_PERSON
    )

    eligible_people = [
        person
        for person in people
        if len(person["images"]) >= required_images
    ]

    print(
        f"People with at least "
        f"{required_images} images: "
        f"{len(eligible_people)}"
    )

    # --------------------------------------------------------
    # Print a few examples
    # --------------------------------------------------------

    print()
    print("Example identities found:")

    for person in eligible_people[:10]:

        print(
            f"  {person['name']}: "
            f"{len(person['images'])} images"
        )

    # --------------------------------------------------------
    # Check sufficient identities
    # --------------------------------------------------------

    required_people = (
        NUM_KNOWN_IDENTITIES
        + NUM_UNKNOWN_IDENTITIES
    )

    if len(eligible_people) < required_people:

        raise RuntimeError(
            f"\nOnly {len(eligible_people)} identities "
            f"have enough images.\n"
            f"We need {required_people}."
        )

    # --------------------------------------------------------
    # Shuffle identities
    # --------------------------------------------------------

    random.shuffle(eligible_people)

    known_people = eligible_people[
        :NUM_KNOWN_IDENTITIES
    ]

    unknown_people = eligible_people[
        NUM_KNOWN_IDENTITIES:
        NUM_KNOWN_IDENTITIES + NUM_UNKNOWN_IDENTITIES
    ]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    GALLERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Split structure
    # --------------------------------------------------------

    split = {
        "seed": RANDOM_SEED,
        "known": {},
        "unknown": {},
    }

    # ========================================================
    # Known identities
    # ========================================================

    for person in known_people:

        images = person["images"].copy()

        random.shuffle(images)

        gallery_images = images[
            :GALLERY_IMAGES_PER_PERSON
        ]

        test_images = images[
            GALLERY_IMAGES_PER_PERSON:
            GALLERY_IMAGES_PER_PERSON
            + TEST_IMAGES_PER_PERSON
        ]

        split["known"][person["name"]] = {
            "gallery": [
                str(
                    image.relative_to(BASE_DIR)
                )
                for image in gallery_images
            ],
            "test": [
                str(
                    image.relative_to(BASE_DIR)
                )
                for image in test_images
            ],
        }

    # ========================================================
    # Unknown identities
    # ========================================================

    for person in unknown_people:

        images = person["images"].copy()

        random.shuffle(images)

        test_images = images[
            :TEST_IMAGES_PER_PERSON
        ]

        split["unknown"][person["name"]] = {
            "test": [
                str(
                    image.relative_to(BASE_DIR)
                )
                for image in test_images
            ]
        }

    # --------------------------------------------------------
    # Save split
    # --------------------------------------------------------

    with open(
        SPLIT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            split,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print()
    print("KNOWN IDENTITIES:")

    for identity, data in split["known"].items():

        print(
            f"  {identity}: "
            f"{len(data['gallery'])} gallery + "
            f"{len(data['test'])} test"
        )

    print()
    print("UNKNOWN IDENTITIES:")

    for identity, data in split["unknown"].items():

        print(
            f"  {identity}: "
            f"{len(data['test'])} test"
        )

    print()
    print("Split saved to:")

    print(SPLIT_FILE)


if __name__ == "__main__":
    main()