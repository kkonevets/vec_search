logconfig_dict = {
    'formatters': {
        'simple': {
            'format': '%(asctime)s [%(process)d] %(message)s',
            'datefmt': '[%Y/%m/%d %H:%M:%S %Z]'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.FileHandler',
            'formatter': 'simple',
            'filename': '../../data/vec_search.logs',
        },
        'error_console': {
            'class': 'logging.FileHandler',
            'formatter': 'simple',
            'filename': '../../data/vec_search.logs',
        },
    },
    'version': 1,
    'disable_existing_loggers': False,
}
