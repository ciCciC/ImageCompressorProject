from dotenv import dotenv_values

config = {
    **dotenv_values(".env"),
    **dotenv_values(".env.test")
}

# DEV
HOST = config.get('HOST', "127.0.0.1")
PORT = config.get('PORT', 5000)
DATABASE_URI = config.get('DATABASE_URI', "sqlite:///data\images_db")

# TEST
DATABASE_TEST_URI = config.get('DATABASE_TEST_URI', "sqlite://")
