from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

# Change to the web directory
os.chdir(os.path.join(os.path.dirname(__file__), 'web'))

# Create server
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
print("Server started at http://localhost:8000")

try:
    server.serve_forever()
except KeyboardInterrupt:
    server.server_close()
    print("\nServer stopped.")