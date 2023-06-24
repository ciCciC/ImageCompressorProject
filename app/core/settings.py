from dotenv import dotenv_values

config = {
    **dotenv_values(".env"),
    **dotenv_values(".env.test")
}

HOST = config.get('HOST')
PORT = config.get('PORT')
DATABASE_URI = config.get('DATABASE_URI')

# TEST
DATABASE_TEST_URI = config.get('DATABASE_TEST_URI')
