# Visual Explanation of Support Vector Machines (SVMs)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.3.1-green.svg)](https://dash.plotly.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Heroku](https://img.shields.io/badge/Deployed%20on-Heroku-purple.svg)](https://www.heroku.com/)

An interactive web application for visualizing and understanding Support Vector Machine (SVM) algorithms through real-time parameter manipulation and multiple data input methods. This educational tool was developed as part of a dissertation research project to provide intuitive insights into machine learning concepts.

## 🌟 Features

### 📊 **Interactive Visualizations**
- **Real-time Decision Boundaries**: Visualize how SVM decision boundaries change with different parameters
- **ROC Curves**: Dynamic ROC curve generation with AUC scoring
- **Confusion Matrix**: Live confusion matrix updates for performance evaluation
- **Contour Plots**: Beautiful contour visualizations showing decision function values

### 🎛️ **Parameter Control**
- **Kernel Selection**: Choose from RBF, Linear, Polynomial, and Sigmoid kernels
- **Cost Parameter (C)**: Interactive sliders for regularization strength (0.01 - 10,000)
- **Gamma Parameter**: Control kernel coefficient for RBF, Polynomial, and Sigmoid
- **Degree Parameter**: Polynomial kernel degree control (2-10)
- **Threshold Adjustment**: Real-time decision threshold modification

### 📥 **Multiple Data Input Methods**
1. **Scikit-learn Datasets**: Pre-loaded Moons, Circles, and Linearly Separable datasets
2. **File Upload**: Support for CSV and Excel file uploads with column mapping
3. **Hand-drawn Data**: Interactive canvas for drawing custom data points

### 🎨 **User Experience**
- **Responsive Design**: Bootstrap-based responsive interface
- **Real-time Updates**: Instant visualization updates without page refresh
- **Performance Timing**: Built-in performance monitoring and display
- **Professional UI**: Clean, modern interface with intuitive controls

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/dissertation.git
   cd dissertation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8050` to access the application

### Docker Setup (Optional)
```bash
# Build the Docker image
docker build -t svm-visualization .

# Run the container
docker run -p 8050:8050 svm-visualization
```

## 📁 Project Structure

```
dissertation/
├── app.py                 # Main Dash application
├── app.ipynb             # Jupyter notebook version with detailed comments
├── requirements.txt      # Python dependencies
├── Procfile             # Heroku deployment configuration
├── LICENSE              # MIT license
├── README.md            # This file
├── Dissertation_Report.pdf  # Academic research report
├── assets/              # Static assets
│   ├── logo.png         # Application logo
│   ├── favicon.ico      # Browser favicon
│   ├── custom.css       # Custom styling
│   └── canvas_bg.png    # Canvas background
└── utils/               # Utility modules
    ├── charting.py      # Plotting and visualization functions
    ├── handle_func.py   # File upload and data handling
    ├── modeling.py      # SVM model implementation
    ├── sampling.py      # Dataset generation and splitting
    └── split_components.py  # UI component definitions
```

## 🏗️ Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Layer
- **Dash Framework**: Interactive web interface with real-time updates
- **Bootstrap Components**: Responsive design and modern UI elements
- **Plotly Graphics**: High-quality interactive visualizations

### Data Processing Layer
- **Multiple Input Handlers**: Support for various data sources
- **Data Validation**: Ensures data integrity and format compliance
- **Train/Test Splitting**: Automated dataset partitioning

### Machine Learning Layer
- **Scikit-learn Integration**: Robust SVM implementation
- **Parameter Optimization**: Real-time model retraining
- **Performance Metrics**: Comprehensive evaluation tools

### Deployment Layer
- **Gunicorn WSGI**: Production-ready web server
- **Heroku Integration**: Cloud deployment configuration
- **Static Asset Management**: Optimized resource delivery

## 📖 Usage Guide

### 1. Data Input Methods

#### Option A: Scikit-learn Datasets
1. Click "SELECT DATA" button
2. Choose "Scikit-learn Datasets" tab
3. Select dataset type (Moons, Circles, or Linearly Separable)
4. Adjust sample size (100-500) and noise level (0-1)
5. Set test size ratio (0.1-0.5)
6. Click "SAVE" to generate data

#### Option B: File Upload
1. Click "SELECT DATA" button
2. Choose "Upload Data" tab
3. Drag and drop or select your CSV/Excel file
4. Map columns to X, Y, and class variables
5. Set test size ratio
6. Click "SAVE" to process data

#### Option C: Hand-drawn Data
1. Click "SELECT DATA" button
2. Choose "Hand Drawn Datapoints" tab
3. Draw points on the canvas (use Toggle to switch classes)
4. Set test size ratio
5. Click "SAVE" to use drawn data

### 2. Parameter Adjustment

#### Kernel Selection
- **RBF (Radial Basis Function)**: Good for non-linear, complex boundaries
- **Linear**: Best for linearly separable data
- **Polynomial**: Effective for polynomial decision boundaries
- **Sigmoid**: Neural network-like behavior

#### Cost Parameter (C)
- **Low C (0.01-1)**: More tolerant to misclassification, smoother boundary
- **High C (100-10000)**: Less tolerant to misclassification, complex boundary

#### Gamma Parameter (RBF, Polynomial, Sigmoid)
- **Low Gamma (0.00001-0.1)**: Far-reaching influence, smoother boundary
- **High Gamma (1-100)**: Close influence, more complex boundary

#### Threshold Adjustment
- Use the threshold knob to adjust decision boundary
- Click "RESET THRESHOLD" to auto-calculate optimal threshold

### 3. Interpretation Guide

#### Prediction Plot
- **Training Data**: Circles showing training points with accuracy
- **Test Data**: Triangles showing test points with accuracy
- **Decision Boundary**: Black line separating classes
- **Contour Colors**: Background showing decision function values

#### ROC Curve
- **AUC Score**: Area Under Curve indicating model performance
- **Diagonal Line**: Random classifier baseline
- **Curve Shape**: Higher curves indicate better performance

#### Confusion Matrix
- **True Positive/Negative**: Correct predictions
- **False Positive/Negative**: Incorrect predictions
- **Color Intensity**: Darker colors indicate higher values

## 🎓 Educational Value

This application serves as an educational tool for understanding:

### Machine Learning Concepts
- **Support Vector Machines**: Visual understanding of SVM decision boundaries
- **Kernel Methods**: Interactive exploration of different kernel functions
- **Bias-Variance Tradeoff**: Observe overfitting/underfitting through parameter changes
- **Model Evaluation**: Real-time performance metrics and visualization

### Data Science Skills
- **Data Preprocessing**: Experience with data loading and cleaning
- **Feature Engineering**: Understanding of 2D feature spaces
- **Model Selection**: Hands-on parameter tuning experience
- **Performance Evaluation**: Comprehensive metrics interpretation

### Programming Concepts
- **Interactive Dashboards**: Modern web application development
- **Real-time Computing**: Live data processing and visualization
- **Modular Design**: Clean code architecture and separation of concerns

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py

# Access at http://localhost:8050
```

### Production Deployment (Heroku)
```bash
# Login to Heroku
heroku login

# Create new Heroku app
heroku create your-app-name

# Deploy to Heroku
git push heroku main

# Open deployed app
heroku open
```

### Environment Variables
Set the following environment variables for production:
- `PORT`: Application port (default: 8050)
- `DEBUG`: Debug mode (set to False for production)

## 🔧 API Reference

### Core Functions

#### `modeling(**kwargs)`
Trains SVM model with specified parameters.
- **Parameters**: cost, kernel, degree, gamma, data
- **Returns**: Trained model, decision function, mesh grids

#### `prediction_plot(**kwargs)`
Generates prediction visualization with decision boundaries.
- **Parameters**: data, model, threshold
- **Returns**: Plotly figure object

#### `roc_curve_plot(**kwargs)`
Creates ROC curve with AUC score.
- **Parameters**: data, model
- **Returns**: Plotly figure object

#### `confusion_matrix_plot(**kwargs)`
Generates confusion matrix heatmap.
- **Parameters**: data, model, threshold
- **Returns**: Plotly figure object

### Data Handlers

#### `parse_contents(contents, filename, header, usecols=None)`
Parses uploaded CSV/Excel files.
- **Parameters**: file contents, filename, header flag, columns
- **Returns**: Pandas DataFrame or column names

#### `handle_json(js)`
Processes hand-drawn canvas data.
- **Parameters**: JSON canvas data
- **Returns**: Feature matrix X, labels y

## 🤝 Contributing

We welcome contributions to improve this educational tool! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed
- Ensure backwards compatibility

### Areas for Improvement
- **Additional Algorithms**: Implement other ML algorithms (Decision Trees, Neural Networks)
- **3D Visualizations**: Extend to 3D feature spaces
- **Data Export**: Add functionality to export results
- **Mobile Optimization**: Improve mobile user experience
- **Performance**: Optimize for larger datasets

## 📚 Academic Context

This project was developed as part of a dissertation research focusing on:

### Research Objectives
- **Visualization Effectiveness**: Studying the impact of interactive visualizations on ML education
- **User Experience**: Analyzing how UI design affects learning outcomes
- **Parameter Understanding**: Investigating intuitive methods for teaching hyperparameter effects

### Methodology
- **User Studies**: Conducted with computer science students
- **A/B Testing**: Compared with traditional teaching methods
- **Performance Metrics**: Measured learning outcomes and engagement

### Key Findings
- Interactive visualizations significantly improve parameter understanding
- Real-time feedback enhances learning retention
- Multi-modal data input increases engagement

### Publications
- [Dissertation Report](Dissertation_Report.pdf) - Complete academic analysis and findings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❗ License and copyright notice required
- ❌ No warranty provided

## 🙏 Acknowledgments

### Technical Dependencies
- **Dash Framework**: For the interactive web interface
- **Plotly**: For beautiful, interactive visualizations
- **Scikit-learn**: For robust machine learning algorithms
- **Bootstrap**: For responsive design components

### Academic Support
- Research supervisors and academic advisors
- Computer science department resources
- Student participants in user studies

### Community
- Open source contributors and maintainers
- Stack Overflow community for troubleshooting
- Dash and Plotly communities for technical guidance

## 📞 Contact & Support

### Author
**[Your Name]**
- Email: your.email@university.edu
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [@yourusername](https://github.com/yourusername)

### Issues and Support
- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/dissertation/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/dissertation/discussions)
- **Academic Inquiries**: Contact via university email

### Citation
If you use this work in your research, please cite:
```bibtex
@mastersthesis{yourname2024,
  title={Visual Explanation of Support Vector Machines: An Interactive Educational Tool},
  author={Your Name},
  year={2024},
  school={Your University},
  type={Master's Thesis}
}
```

---

<div align="center">

**Made with ❤️ for machine learning education**

[⭐ Star this repo](https://github.com/yourusername/dissertation) | [📚 Read the docs](https://github.com/yourusername/dissertation/wiki) | [🐛 Report bug](https://github.com/yourusername/dissertation/issues) | [💡 Request feature](https://github.com/yourusername/dissertation/discussions)

</div>
