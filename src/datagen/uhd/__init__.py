
def is_available():
    try:
        import uhd
        return True
    except ImportError:
        return False

if is_available():
    pass

