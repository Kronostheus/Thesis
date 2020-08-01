class Config:
    DATA_DIR = 'Data/'
    CLASS_DIR = DATA_DIR + 'Classification/'
    SCHEMES_DIR = DATA_DIR + 'Coding_Schemes/'

    TRAIN_DIR = CLASS_DIR + 'Train/'
    TEST_DIR = CLASS_DIR + 'Test/'
    VAL_DIR = CLASS_DIR + 'Val/'

    langs = {'P': 'pt', 'S': 'es', 'I': 'it', 'B': 'pt'}

    random_seed = 42

    MAN_MAJOR_TOPICS = {
        "0": "General",
        "1": "External Relations",
        "2": "Freedom and Democracy",
        "3": "Political System",
        "4": "Economy",
        "5": "Welfare and Quality of Life",
        "6": "Fabric of Society",
        "7": "Social Groups",
        "H": "Header"
    }