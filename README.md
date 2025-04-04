# **Virtual Scholar Agent: Automate Research Paper Retrieval & Summarization**  

The **Virtual Scholar Agent** is an **AI-powered research assistant** that helps researchers, data scientists, and applied scientists stay up-to-date with the latest research papers. It automates the process of **retrieving, evaluating, and summarizing research papers** based on your specific research focus and delivers structured insights via email. 
This project is modified from SesamoHackathon(https://github.com/cc4019/SesamoHackathon), an original idea developed during a hackathon. The current version improves upon the initial concept by refining the workflow, enhancing automation, and optimizing research retrieval and summarization.

## **✨ Features**  
- **Research Analysis**: Uses LLMs to analyze your existing research documents and extract structured insights.  
- **Automated Paper Retrieval**: Searches for relevant papers using the **ArXiv Search Tool** with dynamically generated keywords.  
- **Paper Evaluation**: Uses LLMs to assess the quality and relevance of retrieved papers.  
- **Scheduled Email Summaries**: Sends research paper recommendations via email on a user-defined schedule.  
- **Flexible Automation**: Supports both **local execution on Mac** (via Automator & Calendar Alarm) and **cloud-based execution** (via AWS Lambda & Step Functions).


## **📌 Overall Workflow**  
![Virtual Scholar Agent Workflow](https://github.com/user-attachments/assets/4b0ee1bd-b2a9-4ba9-8de7-217bbd2ac2e0)  

---

## **🛠 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone git@github.com:cc4019/VirtualScholarAgent.git
cd VirtualScholarAgent
```

### **2️⃣ Create and Activate a Virtual Environment**  

**On macOS and Linux:**  
```bash
python3.12 -m venv venv
source venv/bin/activate
```

**On Windows:**  
```bash
python3 -m venv venv
.\venv\Scripts\activate
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up the Centralized Folder**  
Upload your **team documents** (research papers, reports, notes) to the `data/team_docs/` directory for analysis.

### **5️⃣ Configure API Keys & Email Credentials**  

Update the `.env` file with the following credentials:  

- **`OPENAI_API_KEY`** – Used for calling the OpenAI model.  
  - Get your API key from [OpenAI’s API Keys page](https://platform.openai.com/signup/).  

- **`LANGSMITH_API_KEY`** – Used for monitoring the agent workflow.  
  - Sign up at [LangSmith](https://smith.langchain.com/) and generate an API key.  

- **`SENDER_PASSWORD`** – Required for email automation.  
  - Generate an **App Password** for your email at [Google App Passwords](https://myaccount.google.com/apppasswords).  
  - Copy and paste the password into the `.env` file.  

- **Modify Sender & Receiver Emails**  
  - Update the **sender email** and **receiver email** fields in the `.env` file.  

### **6️⃣ Remove Sample Files and Upload Your Own Documents**  

Run the following steps to clean up existing sample files:  

```bash
# Delete the sample research document
rm data/team_docs/sample.pdf

# Remove previously stored analysis and paper retrieval results
rm -rf data/analysis_results/
rm -rf data/paper_retrieval_results/
```

Now, upload your own research documents to `data/team_docs/`.  

### **7️⃣ Start the Virtual Scholar Agent**  

To run the system and receive research updates via email:  
```bash
python src/simple_scheduler.py
```

---

## **🗓 Automating Execution on Mac**  

### **✅ Option 1: Automator & Calendar Alarm (Recommended for Mac Users)**  
1. Open **Finder → Automator**, and create a new **Calendar Alarm**.  
2. Select **Run Shell Script** and paste the following command:  
   ```bash
   python3 /path/to/VirtualScholarAgent/src/simple_scheduler.py
   ```  
3. Save the workflow and schedule the event in **Calendar** to run at your desired frequency.  

### **✅ Option 2: Using `launchd` (Mac’s Job Scheduler)**  
Modify the **`launchd` plist** file and register it with `launchctl` for automated execution.  

### **✅ Option 3: AWS Lambda & Step Functions (Cloud-Based Alternative)**  
If you prefer a **serverless cloud-based solution**, you can:  
- Deploy the script as an **AWS Lambda function**.  
- Use **Amazon EventBridge (CloudWatch Rules)** to schedule execution.  
- Ensure scripts remain within AWS Free Tier limits to avoid charges.  

---

## **📌 References & Further Reading**  
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)  
- [ArXiv API for Paper Retrieval](https://arxiv.org/help/api/index)  
- [Automator & Calendar Alarm Guide](https://support.apple.com/guide/automator/welcome/mac)  

## **🙌 Contributing**  
Contributions are welcome! Feel free to open an issue or submit a pull request.  

## **🐟 License**  
This project is licensed under the **MIT License**.  

---
