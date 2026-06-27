import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sys
# Add parent directory of ui (which is src/) to system path to import logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our model handler
from logic.model_handler import ChurnModelHandler

# Set CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ChurnApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Telco Customer Churn Prediction - ANN Model UI")
        self.geometry("1200x750")
        self.minsize(1050, 680)
        
        # Initialize model handler
        self.model_handler = ChurnModelHandler()
        
        # Layout configuration
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Load dataset statistics for dashboard cards
        self.load_dataset_stats()
        
        # Create UI components
        self.create_sidebar()
        self.create_main_frames()
        
        # Select default view
        self.select_frame("dashboard")
        
        # Flag to indicate if model is training
        self.is_training = False
        
        # Draw charts if model is already trained
        if self.model_handler.is_model_trained():
            self.update_dashboard_with_saved_model()
            
    def load_dataset_stats(self):
        """Loads some quick stats from the CSV to display on the dashboard."""
        self.stats = {
            "total_customers": 0,
            "churn_rate": "0.0%",
            "avg_tenure": "0.0 months",
            "avg_monthly": "$0.00"
        }
        try:
            if os.path.exists(self.model_handler.data_path):
                df = pd.read_csv(self.model_handler.data_path)
                self.stats["total_customers"] = len(df)
                
                churn_count = df['Churn'].value_counts()
                churn_rate_val = (churn_count.get('Yes', 0) / len(df)) * 100
                self.stats["churn_rate"] = f"{churn_rate_val:.1f}%"
                
                self.stats["avg_tenure"] = f"{df['tenure'].mean():.1f} mo"
                self.stats["avg_monthly"] = f"${df['MonthlyCharges'].mean():.2f}"
        except Exception as e:
            print(f"[WARNING] Could not compute stats: {e}")

    # ==========================================
    # SIDEBAR CREATION
    # ==========================================
    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        # Logo / Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="ANN CHURN TOOL", 
            font=ctk.CTkFont(size=20, weight="bold", family="Helvetica")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.sub_logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Artificial Neural Network", 
            text_color="gray",
            font=ctk.CTkFont(size=12, slant="italic")
        )
        self.sub_logo_label.grid(row=1, column=0, padx=20, pady=(0, 30))
        
        # Navigation Buttons
        self.btn_dashboard = ctk.CTkButton(
            self.sidebar_frame, 
            text="Dashboard & Model", 
            anchor="w",
            height=40,
            command=lambda: self.select_frame("dashboard")
        )
        self.btn_dashboard.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_predict = ctk.CTkButton(
            self.sidebar_frame, 
            text="Predict Churn", 
            anchor="w",
            height=40,
            command=lambda: self.select_frame("predict")
        )
        self.btn_predict.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_explorer = ctk.CTkButton(
            self.sidebar_frame, 
            text="Dataset Explorer", 
            anchor="w",
            height=40,
            command=lambda: self.select_frame("explorer")
        )
        self.btn_explorer.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        # Theme controls
        self.theme_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.theme_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.theme_option = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode_event
        )
        self.theme_option.grid(row=6, column=0, padx=20, pady=(5, 20), sticky="ew")

    # ==========================================
    # MAIN FRAMES CREATION
    # ==========================================
    def create_main_frames(self):
        # Create container frame
        self.container_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.container_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.container_frame.grid_rowconfigure(0, weight=1)
        self.container_frame.grid_columnconfigure(0, weight=1)
        
        # Create different views
        self.create_dashboard_frame()
        self.create_predict_frame()
        self.create_explorer_frame()

    def select_frame(self, name):
        # Update sidebar button colors to show active tab
        self.btn_dashboard.configure(fg_color=("#3a7ebf", "#1f538d") if name == "dashboard" else "transparent")
        self.btn_predict.configure(fg_color=("#3a7ebf", "#1f538d") if name == "predict" else "transparent")
        self.btn_explorer.configure(fg_color=("#3a7ebf", "#1f538d") if name == "explorer" else "transparent")
        
        # Hide all frames and show selected one
        if name == "dashboard":
            self.frame_dashboard.grid(row=0, column=0, sticky="nsew")
        else:
            self.frame_dashboard.grid_forget()
            
        if name == "predict":
            self.frame_predict.grid(row=0, column=0, sticky="nsew")
        else:
            self.frame_predict.grid_forget()
            
        if name == "explorer":
            self.frame_explorer.grid(row=0, column=0, sticky="nsew")
            self.load_explorer_table()
        else:
            self.frame_explorer.grid_forget()

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

    # ==========================================
    # VIEW 1: DASHBOARD & TRAINING
    # ==========================================
    def create_dashboard_frame(self):
        self.frame_dashboard = ctk.CTkFrame(self.container_frame, fg_color="transparent")
        self.frame_dashboard.grid_columnconfigure(0, weight=1)
        self.frame_dashboard.grid_columnconfigure(1, weight=1)
        self.frame_dashboard.grid_rowconfigure(1, weight=1)
        
        # --- TITLE & STATUS ---
        self.header_frame = ctk.CTkFrame(self.frame_dashboard, fg_color="transparent", height=60)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.dashboard_title = ctk.CTkLabel(
            self.header_frame, 
            text="Model Dashboard & Neural Network Architecture", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.dashboard_title.grid(row=0, column=0, sticky="w")
        
        # Model status indicator
        status_text = "Status: Trained" if self.model_handler.is_model_trained() else "Status: Untrained"
        status_color = "#2ed573" if self.model_handler.is_model_trained() else "#ff4b4b"
        self.lbl_status = ctk.CTkLabel(
            self.header_frame, 
            text=status_text, 
            text_color=status_color,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.lbl_status.grid(row=0, column=1, sticky="e", padx=10)
        
        # --- LEFT PANEL: ARCHITECTURE & METRICS ---
        self.left_panel = ctk.CTkFrame(self.frame_dashboard)
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Title of architecture
        self.lbl_arch_title = ctk.CTkLabel(
            self.left_panel, 
            text="ANN Architecture (MLPClassifier)", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.lbl_arch_title.grid(row=0, column=0, padx=20, pady=(15, 10), sticky="w")
        
        # Architecture details
        details_txt = (
            "• Hidden Layers: 2 Layers (64 nodes, 32 nodes)\n"
            "• Activation Function: ReLU (Rectified Linear Unit)\n"
            "• Solver Optimizer: Adam\n"
            "• Max Iterations: 500 epochs\n"
            "• Early Stopping: Yes (validation fraction = 10%)"
        )
        self.lbl_arch_details = ctk.CTkLabel(
            self.left_panel, 
            text=details_txt, 
            justify="left", 
            anchor="w",
            font=ctk.CTkFont(size=13)
        )
        self.lbl_arch_details.grid(row=1, column=0, padx=25, pady=(0, 15), sticky="w")
        
        # Dataset Summary Cards (grid inside left panel)
        self.cards_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.cards_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.cards_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Card 1: Total Dataset
        self.card_total = ctk.CTkFrame(self.cards_frame, border_width=1, border_color="#3a3d40", height=80)
        self.card_total.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(self.card_total, text="Total Dataset", text_color="gray", font=ctk.CTkFont(size=11)).pack(pady=(10, 2))
        self.lbl_total_val = ctk.CTkLabel(self.card_total, text=str(self.stats["total_customers"]), font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_total_val.pack(pady=(0, 10))
        
        # Card 2: Churn Rate
        self.card_churn = ctk.CTkFrame(self.cards_frame, border_width=1, border_color="#3a3d40", height=80)
        self.card_churn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(self.card_churn, text="Churn Rate", text_color="gray", font=ctk.CTkFont(size=11)).pack(pady=(10, 2))
        self.lbl_churn_val = ctk.CTkLabel(self.card_churn, text=self.stats["churn_rate"], font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_churn_val.pack(pady=(0, 10))
        
        # Card 3: Avg Tenure
        self.card_tenure = ctk.CTkFrame(self.cards_frame, border_width=1, border_color="#3a3d40", height=80)
        self.card_tenure.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(self.card_tenure, text="Avg Tenure", text_color="gray", font=ctk.CTkFont(size=11)).pack(pady=(10, 2))
        self.lbl_tenure_val = ctk.CTkLabel(self.card_tenure, text=self.stats["avg_tenure"], font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_tenure_val.pack(pady=(0, 10))
        
        # Card 4: Avg Monthly Charges
        self.card_charge = ctk.CTkFrame(self.cards_frame, border_width=1, border_color="#3a3d40", height=80)
        self.card_charge.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(self.card_charge, text="Avg Monthly Charge", text_color="gray", font=ctk.CTkFont(size=11)).pack(pady=(10, 2))
        self.lbl_charge_val = ctk.CTkLabel(self.card_charge, text=self.stats["avg_monthly"], font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_charge_val.pack(pady=(0, 10))
        
        # Model Metrics results
        self.lbl_metrics_title = ctk.CTkLabel(
            self.left_panel, 
            text="Model Performance Metrics", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.lbl_metrics_title.grid(row=3, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.lbl_metrics_val = ctk.CTkLabel(
            self.left_panel, 
            text="Accuracy: Not Trained\nPrecision: -\nRecall: -\nF1-Score: -", 
            justify="left", 
            anchor="w",
            font=ctk.CTkFont(family="Courier", size=13)
        )
        self.lbl_metrics_val.grid(row=4, column=0, padx=25, pady=(0, 15), sticky="w")
        
        # Action Buttons for Training
        self.btn_train = ctk.CTkButton(
            self.left_panel, 
            text="Train / Retrain Model", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            command=self.start_training_thread
        )
        self.btn_train.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="ew")
        
        # Progress indicator
        self.lbl_train_progress = ctk.CTkLabel(
            self.left_panel, 
            text="", 
            text_color="#ffcc00",
            font=ctk.CTkFont(size=12, slant="italic")
        )
        self.lbl_train_progress.grid(row=6, column=0, padx=20, pady=(0, 10))

        # --- RIGHT PANEL: CHARTS ---
        self.right_panel = ctk.CTkFrame(self.frame_dashboard)
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)
        
        self.lbl_charts_title = ctk.CTkLabel(
            self.right_panel, 
            text="Training Visualization", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.lbl_charts_title.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")
        
        # Container for matplotlib canvas
        self.chart_container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.chart_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Initial empty label inside chart container
        self.lbl_empty_chart = ctk.CTkLabel(
            self.chart_container,
            text="Train the model to visualize the training loss curve and confusion matrix.",
            text_color="gray",
            wraplength=350,
            font=ctk.CTkFont(size=14, slant="italic")
        )
        self.lbl_empty_chart.pack(expand=True, fill="both")

    def update_dashboard_with_saved_model(self):
        """Displays saved metrics directly if model was loaded."""
        try:
            # Recompute accuracy by running evaluation again or loading from saved attributes
            # In our case we didn't save historical evaluation, but we can compute it quickly from the dataset!
            # To save time, we will run train() if model needs evaluation stats, 
            # OR we can just check if we can run prediction and state model is loaded.
            # Let's run a quick evaluations to fill the box if needed, or simply write a nice notice.
            self.lbl_metrics_val.configure(
                text="Model loaded successfully!\nClick 'Train / Retrain Model' to recalculate\nand generate training curve plots."
            )
        except Exception as e:
            print(f"Error loading saved metrics: {e}")

    # ==========================================
    # BACKGROUND TRAINING THREAD
    # ==========================================
    def start_training_thread(self):
        if self.is_training:
            return
        
        self.is_training = True
        self.btn_train.configure(state="disabled", text="Training in progress...")
        self.lbl_train_progress.configure(text="Please wait... Preparing data and training neural network.")
        self.lbl_status.configure(text="Status: Training...", text_color="#ffcc00")
        
        # Start thread
        t = threading.Thread(target=self.run_training)
        t.daemon = True
        t.start()

    def run_training(self):
        try:
            metrics = self.model_handler.train()
            
            # Post back to main GUI thread
            self.after(0, lambda: self.training_completed(metrics))
        except Exception as e:
            self.after(0, lambda: self.training_failed(str(e)))

    def training_completed(self, metrics):
        self.is_training = False
        self.btn_train.configure(state="normal", text="Train / Retrain Model")
        self.lbl_train_progress.configure(text="Success! Model trained successfully.", text_color="#2ed573")
        self.lbl_status.configure(text="Status: Trained", text_color="#2ed573")
        
        # Format metric output
        acc = metrics['accuracy']
        rep = metrics['report']
        
        # Accessing classification report dictionary safely (keys are strings representing class labels, e.g. '1')
        churn_metrics = rep.get('1', rep.get('1.0', {}))
        precision_val = churn_metrics.get('precision', 0.0)
        recall_val = churn_metrics.get('recall', 0.0)
        f1_val = churn_metrics.get('f1-score', 0.0)
        
        metric_text = (
            f"Accuracy    : {acc * 100:.2f}%\n"
            f"Precision   : {precision_val * 100:.2f}% (Churn)\n"
            f"Recall      : {recall_val * 100:.2f}% (Churn)\n"
            f"F1-Score    : {f1_val * 100:.2f}% (Churn)\n"
            f"Training Set: {metrics['train_size']} rows\n"
            f"Testing Set : {metrics['test_size']} rows\n"
            f"Epochs Run  : {metrics['n_iter']}"
        )
        self.lbl_metrics_val.configure(text=metric_text)
        
        # Draw plots
        self.draw_training_plots(metrics)

    def training_failed(self, err_msg):
        self.is_training = False
        self.btn_train.configure(state="normal", text="Train / Retrain Model")
        self.lbl_train_progress.configure(text=f"Failed: {err_msg}", text_color="#ff4b4b")
        self.lbl_status.configure(text="Status: Error", text_color="#ff4b4b")
        messagebox.showerror("Training Error", f"An error occurred during training:\n{err_msg}")

    def draw_training_plots(self, metrics):
        # Clear existing elements inside the chart container
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        # Set custom dark style for matplotlib
        is_dark = ctk.get_appearance_mode() == "Dark"
        plt_style = 'dark_background' if is_dark else 'default'
        
        # Re-initialize the style context
        with plt.style.context(plt_style):
            fig = Figure(figsize=(5, 5.5), dpi=100)
            
            # Subplot 1: Loss Curve
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(metrics['loss_curve'], color='#1f77b4', linewidth=2)
            ax1.set_title("Neural Network Training Loss Curve", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Loss (Cross Entropy)")
            ax1.set_xlabel("Epoch / Iteration")
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Subplot 2: Confusion Matrix
            ax2 = fig.add_subplot(2, 1, 2)
            cm = np.array(metrics['confusion_matrix'])
            
            # Draw heatmap
            im = ax2.imshow(cm, cmap='Blues', interpolation='nearest')
            
            # Labels and ticks
            ax2.set_title("Confusion Matrix (Data Uji)", fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax2, shrink=0.7)
            
            tick_marks = np.arange(2)
            ax2.set_xticks(tick_marks)
            ax2.set_yticks(tick_marks)
            ax2.set_xticklabels(['No Churn', 'Churn'])
            ax2.set_yticklabels(['No Churn', 'Churn'])
            
            # Adding annotations inside matrix
            thresh = cm.max() / 2.
            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else ("black" if not is_dark else "white"))
                    
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
            fig.tight_layout()
            
            # Embed matplotlib inside Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==========================================
    # VIEW 2: CHURN PREDICTION FORM
    # ==========================================
    def create_predict_frame(self):
        self.frame_predict = ctk.CTkFrame(self.container_frame, fg_color="transparent")
        self.frame_predict.grid_columnconfigure(0, weight=3) # Form inputs
        self.frame_predict.grid_columnconfigure(1, weight=2) # Prediction results
        self.frame_predict.grid_rowconfigure(0, weight=1)
        
        # --- LEFT PANEL: INPUT FORM (SCROLLABLE) ---
        self.form_panel = ctk.CTkFrame(self.frame_predict)
        self.form_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.form_panel.grid_rowconfigure(1, weight=1)
        self.form_panel.grid_columnconfigure(0, weight=1)
        
        self.form_title = ctk.CTkLabel(
            self.form_panel, 
            text="Customer Characteristics Form", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.form_title.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")
        
        # Scrollable area for forms
        self.scroll_form = ctk.CTkScrollableFrame(self.form_panel, fg_color="transparent")
        self.scroll_form.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.scroll_form.grid_columnconfigure((0, 1), weight=1)
        
        # We store references of variables here
        self.form_vars = {}
        
        row_idx = 0
        
        # --- SECTION 1: DEMOGRAPHICS ---
        self.sec1_label = ctk.CTkLabel(self.scroll_form, text="1. Demographics & Profile", font=ctk.CTkFont(size=13, weight="bold"), text_color="#1f538d")
        self.sec1_label.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        row_idx += 1
        
        # Gender
        self.create_option_menu(self.scroll_form, "gender", "Gender", ["Male", "Female"], row_idx, 0)
        # Senior Citizen
        self.create_option_menu(self.scroll_form, "SeniorCitizen", "Senior Citizen", ["No", "Yes"], row_idx, 1)
        row_idx += 1
        
        # Partner
        self.create_option_menu(self.scroll_form, "Partner", "Partner (Married/Living together)", ["Yes", "No"], row_idx, 0)
        # Dependents
        self.create_option_menu(self.scroll_form, "Dependents", "Dependents (Supported family)", ["No", "Yes"], row_idx, 1)
        row_idx += 1
        
        # --- SECTION 2: SERVICES SUBSCRIPTION ---
        self.sec2_label = ctk.CTkLabel(self.scroll_form, text="2. Subscribed Services", font=ctk.CTkFont(size=13, weight="bold"), text_color="#1f538d")
        self.sec2_label.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=(15, 5), sticky="w")
        row_idx += 1
        
        # Phone Service
        self.create_option_menu(self.scroll_form, "PhoneService", "Phone Service", ["Yes", "No"], row_idx, 0, command=self.on_phone_service_change)
        # Multiple Lines
        self.create_option_menu(self.scroll_form, "MultipleLines", "Multiple Lines", ["No", "Yes", "No phone service"], row_idx, 1)
        row_idx += 1
        
        # Internet Service
        self.create_option_menu(self.scroll_form, "InternetService", "Internet Service Provider (ISP)", ["Fiber optic", "DSL", "No"], row_idx, 0, command=self.on_internet_service_change)
        # Online Security
        self.create_option_menu(self.scroll_form, "OnlineSecurity", "Online Security Addon", ["No", "Yes", "No internet service"], row_idx, 1)
        row_idx += 1
        
        # Online Backup
        self.create_option_menu(self.scroll_form, "OnlineBackup", "Online Backup Addon", ["No", "Yes", "No internet service"], row_idx, 0)
        # Device Protection
        self.create_option_menu(self.scroll_form, "DeviceProtection", "Device Protection Addon", ["No", "Yes", "No internet service"], row_idx, 1)
        row_idx += 1
        
        # Tech Support
        self.create_option_menu(self.scroll_form, "TechSupport", "Tech Support (Priority line)", ["No", "Yes", "No internet service"], row_idx, 0)
        # Streaming TV
        self.create_option_menu(self.scroll_form, "StreamingTV", "Streaming TV Service", ["No", "Yes", "No internet service"], row_idx, 1)
        row_idx += 1
        
        # Streaming Movies
        self.create_option_menu(self.scroll_form, "StreamingMovies", "Streaming Movies Service", ["No", "Yes", "No internet service"], row_idx, 0)
        row_idx += 1
        
        # --- SECTION 3: CONTRACT & CHARGES ---
        self.sec3_label = ctk.CTkLabel(self.scroll_form, text="3. Contract & Charges Billing", font=ctk.CTkFont(size=13, weight="bold"), text_color="#1f538d")
        self.sec3_label.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=(15, 5), sticky="w")
        row_idx += 1
        
        # Contract
        self.create_option_menu(self.scroll_form, "Contract", "Contract Duration", ["Month-to-month", "One year", "Two year"], row_idx, 0)
        # Paperless Billing
        self.create_option_menu(self.scroll_form, "PaperlessBilling", "Paperless Billing", ["Yes", "No"], row_idx, 1)
        row_idx += 1
        
        # Payment Method
        self.create_option_menu(self.scroll_form, "PaymentMethod", "Payment Method", 
                                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 
                                row_idx, 0, columnspan=2)
        row_idx += 1
        
        # Tenure Slider
        self.create_slider_control(self.scroll_form, "tenure", "Tenure (Customer Months)", 0, 72, 12, "%d months", row_idx, 0, columnspan=2, callback=self.update_charges)
        row_idx += 1
        
        # Monthly Charges Slider
        self.create_slider_control(self.scroll_form, "MonthlyCharges", "Monthly Charges ($)", 18.0, 120.0, 50.0, "$%.2f", row_idx, 0, columnspan=2, callback=self.update_charges)
        row_idx += 1
        
        # Total Charges (Calculated with manual override entry)
        self.lbl_tot_charge_title = ctk.CTkLabel(self.scroll_form, text="Total Charges ($) (Auto-calculated / Override):", anchor="w")
        self.lbl_tot_charge_title.grid(row=row_idx, column=0, padx=10, pady=(10, 2), sticky="w")
        
        self.entry_total_charges = ctk.CTkEntry(self.scroll_form)
        self.entry_total_charges.grid(row=row_idx+1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        row_idx += 2
        
        # Initialize calculations
        self.update_charges(None)

        # --- RIGHT PANEL: PREDICTION RESULTS ---
        self.result_panel = ctk.CTkFrame(self.frame_predict)
        self.result_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        self.result_panel.grid_columnconfigure(0, weight=1)
        self.result_panel.grid_rowconfigure(2, weight=1)
        
        # Main predict action button
        self.btn_predict_run = ctk.CTkButton(
            self.result_panel, 
            text="RUN CHURN PREDICTION", 
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#2ed573",
            hover_color="#26af5f",
            height=50,
            command=self.execute_prediction
        )
        self.btn_predict_run.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        # Result Header
        self.lbl_result_header = ctk.CTkLabel(
            self.result_panel, 
            text="Prediction Output Card", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.lbl_result_header.grid(row=1, column=0, padx=20, pady=(5, 10), sticky="w")
        
        # Card Background
        self.result_card = ctk.CTkFrame(self.result_panel, corner_radius=12, fg_color="#1a1c1e")
        self.result_card.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        
        # Inside Result Card
        self.lbl_card_title = ctk.CTkLabel(
            self.result_card, 
            text="READY TO PREDICT", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="gray"
        )
        self.lbl_card_title.pack(pady=(35, 10))
        
        self.lbl_churn_verdict = ctk.CTkLabel(
            self.result_card, 
            text="Input details & click predict", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.lbl_churn_verdict.pack(pady=(0, 20))
        
        # Probability Display
        self.lbl_prob_title = ctk.CTkLabel(
            self.result_card, 
            text="Churn Probability:", 
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.lbl_prob_title.pack(pady=(10, 2))
        
        self.lbl_prob_percent = ctk.CTkLabel(
            self.result_card, 
            text="--%", 
            font=ctk.CTkFont(size=36, weight="bold")
        )
        self.lbl_prob_percent.pack(pady=(0, 15))
        
        # Horizontal progress bar
        self.prob_bar = ctk.CTkProgressBar(self.result_card, width=220)
        self.prob_bar.pack(pady=(0, 25))
        self.prob_bar.set(0.0)
        
        # Tips / Details explaining recommendation
        self.lbl_rec_title = ctk.CTkLabel(
            self.result_card, 
            text="Recommendation / Risk Level:", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_rec_title.pack(pady=(10, 2), padx=20, anchor="w")
        
        self.lbl_recommendation = ctk.CTkLabel(
            self.result_card, 
            text="Enter customer characteristics to analyze risk factors.", 
            wraplength=250,
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray"
        )
        self.lbl_recommendation.pack(pady=(0, 30), padx=20)

    # --- HELPER UI FUNCTIONS ---
    def create_option_menu(self, master, var_name, label_text, values, row, column, columnspan=1, command=None):
        frame = ctk.CTkFrame(master, fg_color="transparent")
        frame.grid(row=row, column=column, columnspan=columnspan, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        
        lbl = ctk.CTkLabel(frame, text=label_text, anchor="w")
        lbl.grid(row=0, column=0, pady=(0, 2), sticky="w")
        
        self.form_vars[var_name] = ctk.StringVar(value=values[0])
        opt = ctk.CTkOptionMenu(frame, values=values, variable=self.form_vars[var_name], command=command)
        opt.grid(row=1, column=0, sticky="ew")
        return opt

    def create_slider_control(self, master, var_name, label_text, from_val, to_val, start_val, format_str, row, column, columnspan=1, callback=None):
        frame = ctk.CTkFrame(master, fg_color="transparent")
        frame.grid(row=row, column=column, columnspan=columnspan, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        
        # Create a container header for label and actual value text
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 2))
        
        lbl = ctk.CTkLabel(header, text=label_text, anchor="w")
        lbl.pack(side="left")
        
        lbl_val = ctk.CTkLabel(header, text=format_str % start_val, font=ctk.CTkFont(weight="bold"))
        lbl_val.pack(side="right")
        
        self.form_vars[var_name] = ctk.DoubleVar(value=start_val)
        
        # Slide event
        def on_slide(val):
            # Parse to int or float based on variable type
            if "%d" in format_str:
                val = int(float(val))
                lbl_val.configure(text=format_str % val)
            else:
                val = float(val)
                lbl_val.configure(text=format_str % val)
                
            if callback:
                callback(val)
                
        slider = ctk.CTkSlider(frame, from_=from_val, to=to_val, variable=self.form_vars[var_name], command=on_slide)
        slider.pack(fill="x")
        return slider

    def update_charges(self, _):
        """Calculates TotalCharges as tenure * MonthlyCharges and updates entry."""
        try:
            tenure = int(float(self.form_vars['tenure'].get()))
            monthly = float(self.form_vars['MonthlyCharges'].get())
            total = tenure * monthly
            
            # Set entry value
            self.entry_total_charges.delete(0, tk.END)
            self.entry_total_charges.insert(0, f"{total:.2f}")
        except Exception:
            pass

    # --- DYNAMIC FORM HANDLERS (Interlocking choices) ---
    def on_phone_service_change(self, value):
        # If PhoneService is No, MultipleLines must be "No phone service"
        if value == "No":
            self.form_vars["MultipleLines"].set("No phone service")
        elif self.form_vars["MultipleLines"].get() == "No phone service":
            self.form_vars["MultipleLines"].set("No")

    def on_internet_service_change(self, value):
        # If InternetService is No, OnlineSecurity etc. must be "No internet service"
        internet_addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        if value == "No":
            for addon in internet_addons:
                self.form_vars[addon].set("No internet service")
        else:
            for addon in internet_addons:
                # If they were set to 'No internet service', switch to 'No'
                if self.form_vars[addon].get() == "No internet service":
                    self.form_vars[addon].set("No")

    # --- EXECUTE PREDICTION ---
    def execute_prediction(self):
        if not self.model_handler.is_model_trained():
            messagebox.showwarning("Model Not Trained", "The ANN model must be trained first. Go to 'Dashboard & Model' tab and train the model.")
            return

        try:
            # Build input dictionary
            inputs = {
                'gender': self.form_vars['gender'].get(),
                'SeniorCitizen': self.form_vars['SeniorCitizen'].get(),
                'Partner': self.form_vars['Partner'].get(),
                'Dependents': self.form_vars['Dependents'].get(),
                'tenure': int(float(self.form_vars['tenure'].get())),
                'PhoneService': self.form_vars['PhoneService'].get(),
                'MultipleLines': self.form_vars['MultipleLines'].get(),
                'InternetService': self.form_vars['InternetService'].get(),
                'OnlineSecurity': self.form_vars['OnlineSecurity'].get(),
                'OnlineBackup': self.form_vars['OnlineBackup'].get(),
                'DeviceProtection': self.form_vars['DeviceProtection'].get(),
                'TechSupport': self.form_vars['TechSupport'].get(),
                'StreamingTV': self.form_vars['StreamingTV'].get(),
                'StreamingMovies': self.form_vars['StreamingMovies'].get(),
                'Contract': self.form_vars['Contract'].get(),
                'PaperlessBilling': self.form_vars['PaperlessBilling'].get(),
                'PaymentMethod': self.form_vars['PaymentMethod'].get(),
                'MonthlyCharges': float(self.form_vars['MonthlyCharges'].get()),
                'TotalCharges': float(self.entry_total_charges.get().strip())
            }
            
            # Predict
            result = self.model_handler.predict(inputs)
            
            # Update UI
            churn = result['churn_prediction']
            prob = result['churn_probability']
            
            self.lbl_prob_percent.configure(text=f"{prob * 100:.1f}%")
            self.prob_bar.set(prob)
            
            if churn == 'Yes':
                self.lbl_card_title.configure(text="HIGH CHURN RISK DETECTED", text_color="#ff4b4b")
                self.lbl_churn_verdict.configure(text="CUSTOMER WILL LIKELY LEAVE")
                self.result_card.configure(fg_color="#3e1b1b") # Dark red bg
                
                # Recommendation advice based on key risk factors
                rec_text = "Action Needed:\n"
                if inputs['Contract'] == 'Month-to-month':
                    rec_text += "• High Risk: Month-to-month contract. Offer 1-year or 2-year contract discount.\n"
                if inputs['InternetService'] == 'Fiber optic':
                    rec_text += "• Fiber optic internet detected. Check if they have tech support issues.\n"
                if inputs['TechSupport'] == 'No':
                    rec_text += "• Offer complementary priority Tech Support service.\n"
                if inputs['tenure'] < 12:
                    rec_text += "• Early customer (under 1 year). Send engagement promo code."
                
                self.lbl_recommendation.configure(text=rec_text, text_color="#ffb3b3")
            else:
                self.lbl_card_title.configure(text="LOW CHURN RISK", text_color="#2ed573")
                self.lbl_churn_verdict.configure(text="CUSTOMER IS STABLE")
                self.result_card.configure(fg_color="#1b331e") # Dark green bg
                
                rec_text = "Loyalty Status:\n"
                if inputs['Contract'] != 'Month-to-month':
                    rec_text += "• Healthy long-term contract active.\n"
                if inputs['OnlineSecurity'] == 'Yes' or inputs['TechSupport'] == 'Yes':
                    rec_text += "• High value added: Multiple supportive services subscribed.\n"
                rec_text += "• Maintain current promotional and service levels."
                
                self.lbl_recommendation.configure(text=rec_text, text_color="#b3ffb3")
                
        except ValueError as ve:
            messagebox.showerror("Input Error", f"Total Charges must be a valid number. Error: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

    # ==========================================
    # VIEW 3: DATASET EXPLORER
    # ==========================================
    def create_explorer_frame(self):
        self.frame_explorer = ctk.CTkFrame(self.container_frame, fg_color="transparent")
        self.frame_explorer.grid_columnconfigure(0, weight=1)
        self.frame_explorer.grid_rowconfigure(1, weight=1)
        
        # Title & Description
        self.explorer_header = ctk.CTkFrame(self.frame_explorer, fg_color="transparent", height=50)
        self.explorer_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.explorer_title = ctk.CTkLabel(
            self.explorer_header, 
            text="Telco Customer Churn Dataset Explorer", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.explorer_title.pack(side="left", anchor="w")
        
        self.lbl_row_count = ctk.CTkLabel(
            self.explorer_header, 
            text="Displaying first 50 rows of data", 
            text_color="gray"
        )
        self.lbl_row_count.pack(side="right", anchor="e")
        
        # Table frame to hold Treeview
        self.table_frame = ctk.CTkFrame(self.frame_explorer)
        self.table_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.table_frame.grid_columnconfigure(0, weight=1)
        self.table_frame.grid_rowconfigure(0, weight=1)
        
        # We will dynamically populate the Treeview in load_explorer_table

    def load_explorer_table(self):
        # Clear existing elements inside the table frame first to avoid duplication
        for widget in self.table_frame.winfo_children():
            widget.destroy()
            
        try:
            if not os.path.exists(self.model_handler.data_path):
                err_lbl = ctk.CTkLabel(self.table_frame, text="Dataset file not found!", text_color="#ff4b4b")
                err_lbl.pack(expand=True)
                return
                
            df = pd.read_csv(self.model_handler.data_path)
            
            # Select first 50 rows and a subset of columns for clean rendering
            preview_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                            'tenure', 'PhoneService', 'InternetService', 'Contract', 
                            'MonthlyCharges', 'TotalCharges', 'Churn']
            df_preview = df[preview_cols].head(50)
            
            # Create a styled Treeview (Tkinter standard with styling)
            style = ttk.Style()
            
            # Detect dark vs light theme to match styles
            is_dark = ctk.get_appearance_mode() == "Dark"
            bg_color = "#1e1e1e" if is_dark else "#ffffff"
            fg_color = "#ffffff" if is_dark else "#000000"
            field_bg = "#2b2b2b" if is_dark else "#f0f0f0"
            select_bg = "#1f538d" if is_dark else "#3a7ebf"
            
            style.theme_use("clam")
            style.configure("Treeview", 
                            background=bg_color, 
                            foreground=fg_color, 
                            fieldbackground=field_bg,
                            rowheight=25,
                            gridcolor="#3a3d40" if is_dark else "#dcdcdc")
            style.map("Treeview", background=[("selected", select_bg)])
            
            style.configure("Treeview.Heading", 
                            background="#343638" if is_dark else "#e0e0e0", 
                            foreground=fg_color,
                            relief="flat")
            
            # Scrollbars
            tree_scroll_y = ctk.CTkScrollbar(self.table_frame, orientation="vertical")
            tree_scroll_y.pack(side="right", fill="y")
            
            tree_scroll_x = ctk.CTkScrollbar(self.table_frame, orientation="horizontal")
            tree_scroll_x.pack(side="bottom", fill="x")
            
            # Treeview widget
            self.tree = ttk.Treeview(
                self.table_frame, 
                columns=preview_cols, 
                show="headings",
                yscrollcommand=tree_scroll_y.set,
                xscrollcommand=tree_scroll_x.set
            )
            self.tree.pack(fill="both", expand=True)
            
            # Link scrollbars
            tree_scroll_y.configure(command=self.tree.yview)
            tree_scroll_x.configure(command=self.tree.xview)
            
            # Columns configuration
            for col in preview_cols:
                # Custom width
                width = 90
                if col == 'customerID':
                    width = 110
                elif col in ['MonthlyCharges', 'TotalCharges']:
                    width = 100
                elif col == 'PaymentMethod':
                    width = 160
                self.tree.heading(col, text=col, anchor="w")
                self.tree.column(col, width=width, anchor="w")
            
            # Insert values
            for idx, row in df_preview.iterrows():
                self.tree.insert("", "end", values=list(row))
                
        except Exception as e:
            err_lbl = ctk.CTkLabel(self.table_frame, text=f"Error displaying dataset:\n{e}", text_color="#ff4b4b")
            err_lbl.pack(expand=True)

if __name__ == "__main__":
    app = ChurnApp()
    app.mainloop()
