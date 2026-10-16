# 🦜🔗 LangChain Core Practical

A hands-on practical demonstration of every core **LangChain** component. Each topic has its own folder with runnable Python files, so you can test every model, chain, component, and agent one by one directly from the terminal.

---

## 📌 What's Inside

- Separate folder for each LangChain topic/component
- Each folder contains standalone `.py` files you can run individually
- Uses **Groq** (fast LLM inference), **Hugging Face** (models/embeddings), and **WeatherStack** (live weather data for the weather agent)
- A `files/` folder for your own **RAG** documents (PDF and TXT)
- A **complete Theory documentation** in easy language with explanation

---

## 📂 Project Structure

```
LangChain-Core-Practical/
│
├── .env                  # API keys (you create this - never share it)
├── requirements.txt      # All Python dependencies
├── README.md
│
├── <topic_folder_1>/     # One folder per LangChain topic
│   └── <file_name>.py
│
├── <topic_folder_2>/
│   └── <file_name>.py
│
├── ...
│
└── files/                # RAG base documents (add your own PDF and TXT here)
    ├── your_document.pdf
    └── your_document.txt
```

---

## ✅ Prerequisites

- **Python 3.10 or higher** → [Download Python](https://www.python.org/downloads/)
- **Git** → [Download Git](https://git-scm.com/downloads)
- A code editor such as [VS Code](https://code.visualstudio.com/) (optional but recommended)

Check your installs:

```bash
python --version
git --version
```

---

## 🚀 Step 1: Clone the Repository

```bash
git clone https://github.com/RiteshPatil2001/LangChain-Core-Practical.git
cd LangChain-Core-Practical
```

---

## 🧪 Step 2: Create a Virtual Environment (Recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Step 4: Get Your API Keys

You need three API keys. All of them have a free tier.

### 4.1 Groq API Key

1. Go to the Groq Console: **https://console.groq.com/**
2. Sign up or log in.
3. Open the API Keys page: **https://console.groq.com/keys**
4. Click **Create API Key**, give it a name, and click **Submit**.
5. **Copy the key immediately.** It is shown only once.

### 4.2 Hugging Face API Key (Access Token)

1. Go to Hugging Face: **https://huggingface.co/join** (sign up) or **https://huggingface.co/login** (log in).
2. Open the Access Tokens page: **https://huggingface.co/settings/tokens**
3. Click **Create new token**.
4. Choose a token type (**Read** is enough for most demos), give it a name, and click **Create token**.
5. **Copy the token** and save it somewhere safe.

### 4.3 WeatherStack API Key (for the Weather Agent)

1. Go to WeatherStack: **https://weatherstack.com/**
2. Click **Sign Up** and choose the **Free** plan: **https://weatherstack.com/signup/free**
3. Complete registration and verify your email.
4. Open your dashboard: **https://weatherstack.com/dashboard**
5. **Copy your API Access Key** from the dashboard.

---

## 🔐 Step 5: Create the `.env` File

In the **root folder** of the project (the same place as `requirements.txt`), create a file named `.env` and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

> ⚠️ Replace each placeholder with your real key.
> ⚠️ Add WEATHERSTACK_API_KEY in 'Agents\Custom_Agent.py' file in line no 15.
> 🚫 **Never commit your `.env` file to GitHub.** Make sure `.env` is listed in your `.gitignore`.

---

## 📄 Step 6: Add Your Documents for RAG

The RAG demos need base documents to read from. Because of personal data, the original files are **not included** in this repository.

1. Open the `files/` folder in the project.
2. Add your own **PDF** file and **TXT** file.
3. Open the files where these file used and **replace the file names** in the code with your own file names.

---

## ▶️ Step 7: Run Each File Individually

Every `.py` file is a standalone demo. Run them one by one to verify that each model, component, or agent works correctly.

**Option A: Run from the project root**

```bash
python folder_name/file_name.py
```

**Option B: Go into the topic folder and run the file**

```bash
cd folder_name
python file_name.py
```

**Example**

```bash
python chains/simple_chain.py
```

On macOS/Linux, use `python3` instead of `python` if `python` is not recognized.

### ✔️ How to check that a file works

- The script runs and prints output in the terminal with **no errors**.
- If you see an error, check the section below.

---

## 🧰 Tech Stack

- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [Hugging Face](https://huggingface.co/)
- [WeatherStack](https://weatherstack.com/)
- Python

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to fork the repo and open a pull request.

---

## 👤 Author

**Ritesh Patil**
GitHub: [@RiteshPatil2001](https://github.com/RiteshPatil2001)

---

⭐ If you found this repository helpful, consider giving it a star!
