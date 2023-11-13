if __name__ == '__main__':
    import fastwsgi
    from app import app

    fastwsgi.run(
        wsgi_app=app,
        host='0.0.0.0',
        port=8080
    )
    
