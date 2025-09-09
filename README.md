# 🎓 AI-based Drop-out Prediction and Counseling System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com/)

An intelligent early-warning system that helps educational institutions identify at-risk students and facilitate timely interventions through data-driven insights and secure counseling workflows.

## 📌 Overview

Traditional academic monitoring systems only reveal student struggles after term-end results, when intervention opportunities have often passed. This system provides **real-time risk assessment** by analyzing multiple data streams and flagging students who need immediate attention.

### Key Problem Solved
- **Early Detection**: Identifies struggling students before they disengage completely
- **Data Integration**: Consolidates scattered information from multiple Excel sheets
- **Actionable Insights**: Provides clear visual indicators for educators
- **Secure Counseling**: Ensures accountability and safety in mentor-student interactions

## 🎯 Features

### 📊 **Unified Data Dashboard**
- Merges attendance, academic performance, and fee payment data
- Real-time visualization of student progress trends
- Intuitive color-coded risk indicators

### 🚦 **Risk Assessment System**
- **🔴 High Risk**: Students requiring immediate intervention
- **🟡 Moderate Risk**: Students showing concerning patterns
- **🟢 Safe**: Students performing within acceptable parameters

### 🤖 **ML-Powered Predictions**
- Advanced machine learning models identify hidden patterns
- Predictive analytics beyond simple rule-based thresholds
- Continuous model improvement through feedback loops

### 📩 **Automated Alert System**
- Regular notifications to mentors and guardians
- Customizable alert thresholds and frequencies
- Email and SMS integration for critical cases

### 🎥 **Secure Counseling Framework**
- Mandatory video upload requirement for all counseling sessions
- Administrative oversight for student safety and accountability
- Role-based access control system

## 🏗️ Project Structure

```
Hackathon/
│
├── login/                   # Authentication system
│   ├── index.html          # Landing page
│   ├── login.html          # Login interface
│   ├── login.js            # Authentication logic
│   └── login.css           # Login styling
│
├── admin/                   # Administrative dashboard
│   ├── admin.html          # Admin interface
│   ├── admin.js            # Admin functionality
│   └── admin.css           # Admin styling
│
├── mentor/                  # Mentor dashboard
│   ├── mentor.html         # Mentor interface
│   ├── mentor.js           # Mentor tools & video upload
│   └── mentor.css          # Mentor styling
│
├── student/                 # Student portal
│   ├── student.html        # Student dashboard
│   ├── student.js          # Student insights
│   └── student.css         # Student styling
│
├── data/                    # Data storage directory
│   ├── attendance.xlsx     # Attendance records
│   ├── marks.xlsx          # Academic performance
│   └── fees.xlsx           # Fee payment data
│
├── models/                  # ML models and training
│   ├── trained_model.pkl   # Saved prediction model
│   └── model_config.json   # Model configuration
│
├── uploads/                 # Video uploads storage
│
├── app.py                  # Main Flask application
├── model.py                # ML model implementation
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
├── README.md              # Project documentation
└── .gitignore             # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SanjayChoudhari/AI-based-drop-out-prediction-and-counseling-system.git
   cd AI-based-drop-out-prediction-and-counseling-system
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place your XLSX files in the `data/` directory:
     - `attendance.xlsx` - Student attendance records
     - `marks.xlsx` - Academic performance data
     - `fees.xlsx` - Fee payment information
   
5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the system**
   - Open your browser and navigate to `http://localhost:5000`
   - Use the appropriate login credentials for your role

## 🛠️ Technology Stack

### Backend
- **Python Flask**: Web framework
- **Scikit-learn**: Machine learning
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **OpenPyXL**: Excel file processing

### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript**: Interactive functionality
- **Bootstrap**: Responsive design

### Data Storage
- **Excel/XLSX**: Primary data source
- **Local/Cloud Storage**: Video uploads
- **SQLite**: Session management (optional)

## 🔒 Security & Privacy

### Data Protection
- Role-based access control
- Encrypted data transmission
- Secure file upload handling
- Regular security audits

### Video Monitoring System
- **Mandatory Upload**: All counseling sessions must be recorded
- **Administrative Review**: Videos monitored by authorized personnel
- **Student Safety**: Ensures appropriate mentor-student interactions
- **Transparency**: Clear documentation of all interventions

## 📈 Risk Indicators

The system monitors multiple factors to assess dropout risk:

### Academic Performance
- Declining test scores
- Failed assignments
- Subject-specific struggles

### Behavioral Patterns
- Attendance irregularities
- Class participation levels
- Engagement metrics

### Socio-Economic Factors
- Fee payment delays
- Family background data
- External support systems

## 🎯 Impact & Benefits

### For Educational Institutions
- **Reduced Dropout Rates**: Early intervention prevents student loss
- **Resource Optimization**: Targeted support where needed most
- **Data-Driven Decisions**: Evidence-based educational strategies

### For Students
- **Timely Support**: Help arrives before crisis points
- **Personalized Attention**: Tailored intervention strategies
- **Better Outcomes**: Improved academic success rates

### For Educators
- **Clear Insights**: Visual dashboards highlight priorities
- **Efficient Workflow**: Automated alerts save time
- **Professional Safety**: Video documentation protects all parties

## 🤝 Contributing

We welcome contributions from the community! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines for Python code
- Include unit tests for new features
- Update documentation as needed
- Ensure cross-browser compatibility for frontend changes

## 📞 Support & Contact

### Project Maintainer
**Sanjay Choudhari**
- GitHub: [@SanjayChoudhari](https://github.com/SanjayChoudhari)
- Email: [sanjaychoudhari288@gmail.com]

### Team
**The Debug Society**
- Team Name: The Debug Society. (TL)
- Repository: [AI-based-drop-out-prediction-and-counseling-system](https://github.com/SanjayChoudhari/AI-based-drop-out-prediction-and-counseling-system)

### Getting Help
- 📖 Check the [Wiki](wiki) for detailed documentation
- 🐛 Report bugs via [Issues](issues)
- 💡 Request features through [Feature Requests](issues/new?template=feature_request.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Thanks to educational institutions providing valuable feedback
- Special recognition to mentors who prioritize student welfare
- Appreciation for the open-source community's contributions

---

### 🚀 Ready to Transform Student Success?

This system represents a practical approach to educational technology—leveraging existing data to create meaningful impact without requiring expensive infrastructure. Join us in making education more responsive and student-centered.

**Star ⭐ this repository if you find it useful!**
