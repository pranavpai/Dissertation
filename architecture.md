# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          SVM Visualization Web Application                           │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                            Frontend Layer (Dash)                            │   │
│  │                                                                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │   │
│  │  │   Data Input    │  │   Parameter     │  │    Visualization Panel     │  │   │
│  │  │     Panel       │  │   Controls      │  │                             │  │   │
│  │  │                 │  │                 │  │  ┌─────────────────────────┐ │  │   │
│  │  │ • Dataset Tab   │  │ • Threshold     │  │  │   Prediction Plot       │ │  │   │
│  │  │ • Upload Tab    │  │ • Kernel Type   │  │  │   (Contour + Scatter)   │ │  │   │
│  │  │ • Canvas Tab    │  │ • Cost (C)      │  │  └─────────────────────────┘ │  │   │
│  │  │                 │  │ • Degree        │  │                             │  │   │
│  │  └─────────────────┘  │ • Gamma         │  │  ┌─────────────────────────┐ │  │   │
│  │                       │                 │  │  │      ROC Curve          │ │  │   │
│  │                       └─────────────────┘  │  └─────────────────────────┘ │  │   │
│  │                                            │                             │  │   │
│  └─────────────────────────────────────────────┘  ┌─────────────────────────┐ │  │   │
│                                                   │   Confusion Matrix      │ │  │   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │  │   │
│  │                        Data Processing Layer                            │ │  │   │
│  │                                                                         │ │  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  └─┘  │   │
│  │  │   Data Sources  │  │  Data Handlers  │  │     Data Splitters      │      │   │
│  │  │                 │  │                 │  │                         │      │   │
│  │  │ • Scikit-learn  │  │ • CSV Parser    │  │ • Train/Test Split      │      │   │
│  │  │   Datasets      │  │ • Excel Parser  │  │ • Random State Control │      │   │
│  │  │   - Moons       │  │ • JSON Handler  │  │                         │      │   │
│  │  │   - Circles     │  │   (Canvas)      │  │                         │      │   │
│  │  │   - Linear Sep. │  │                 │  │                         │      │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                      Machine Learning Layer                                 │   │
│  │                                                                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │   │
│  │  │  SVM Modeling   │  │   Evaluation    │  │      Visualization          │  │   │
│  │  │                 │  │                 │  │                             │  │   │
│  │  │ • SVC Algorithm │  │ • Accuracy      │  │ • Decision Boundaries       │  │   │
│  │  │ • Kernel Types: │  │ • ROC-AUC       │  │ • Contour Plots             │  │   │
│  │  │   - RBF         │  │ • Confusion     │  │ • Scatter Plots             │  │   │
│  │  │   - Linear      │  │   Matrix        │  │ • Interactive Thresholds    │  │   │
│  │  │   - Polynomial  │  │                 │  │                             │  │   │
│  │  │   - Sigmoid     │  │                 │  │                             │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Infrastructure Layer                                 │   │
│  │                                                                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │   │
│  │  │   Web Server    │  │   Deployment    │  │       Dependencies          │  │   │
│  │  │                 │  │                 │  │                             │  │   │
│  │  │ • Dash Server   │  │ • Heroku        │  │ • Dash Framework            │  │   │
│  │  │ • Gunicorn      │  │ • Procfile      │  │ • Plotly Graphing           │  │   │
│  │  │ • Flask Backend │  │                 │  │ • Scikit-learn              │  │   │
│  │  │                 │  │                 │  │ • NumPy/Pandas              │  │   │
│  │  │                 │  │                 │  │ • Bootstrap Components      │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘

Data Flow:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ User Input  │───▶│ Data        │───▶│ SVM Model   │───▶│ Predictions │───▶│ Interactive │
│ (3 Methods) │    │ Processing  │    │ Training    │    │ & Metrics   │    │ Visualiza-  │
│             │    │             │    │             │    │             │    │ tion        │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                   │                   │                   │                   │
      │                   │                   │                   │                   │
   Dataset            Train/Test           Decision           ROC Curve         Real-time
   Upload              Split             Function          Confusion Matrix     Updates
  Canvas Draw                                                                      
```

## Component Details

### Frontend Layer
- **Interactive UI**: Built with Dash and Bootstrap components
- **Real-time Updates**: Parameter changes trigger immediate visualization updates
- **Multi-input Support**: Three different data input methods

### Data Processing Layer
- **Multiple Data Sources**: Scikit-learn datasets, file uploads, hand-drawn data
- **Flexible Parsing**: Handles CSV, Excel, and JSON formats
- **Data Validation**: Ensures data integrity before processing

### Machine Learning Layer
- **SVM Implementation**: Uses scikit-learn's SVC with customizable parameters
- **Multiple Kernels**: RBF, Linear, Polynomial, and Sigmoid options
- **Performance Metrics**: Accuracy, ROC-AUC, and confusion matrix evaluation

### Infrastructure Layer
- **Web Deployment**: Gunicorn WSGI server for production
- **Cloud Ready**: Configured for Heroku deployment
- **Responsive Design**: Bootstrap-based responsive interface