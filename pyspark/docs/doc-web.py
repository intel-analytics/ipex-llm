import web

urls = (
    '/(.*)', 'router'
)

app = web.application(urls,globals())

class router:
    def GET(self, path):
        if path == '': path = 'index.html'
        f = open('_build/html/'+path) 
        return f.read()

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
