from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 Flask is running!"

if __name__ == "__main__":
    import os
    print("✅ Running Flask...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
