import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# === FILE PATH ===
csv_path = "/Users/anushaseshadri/Predictive-Anomaly-Detection-in-NGAFID-Data-Using-Neuroevolution-and-Deep-Learning/dataset/before/open_2017_05_08_close_2017_05_08_flight_Fixed Wing_N550ND_before_2_189656.csv"  # ⬅️ Replace this with your actual file path

# === AUTO-DETECT HEADER ===
def find_header_line(path):
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            if not line.startswith("#"):
                return i
    return 0

header_line = find_header_line(csv_path)

# === LOAD FILE ===
df = pd.read_csv(csv_path, skiprows=header_line, low_memory=False)
df.columns = df.columns.str.strip()  # 💡 Fix weird spaces in column names

# === Combine Date and Time ===
if 'Lcl Date' in df.columns and 'Lcl Time' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Lcl Date'] + ' ' + df['Lcl Time'])
else:
    print("⚠️ Missing 'Lcl Date' or 'Lcl Time' — using index as time.")
    df['Timestamp'] = pd.RangeIndex(start=0, stop=len(df), step=1)

# === Plot 1: Altitude vs Time ===
if 'AltMSL' in df.columns:
    plt.figure()
    plt.plot(df["Timestamp"], df["AltMSL"])
    plt.title("Altitude (MSL) vs Time")
    plt.xlabel("Time")
    plt.ylabel("Altitude (ft)")
    plt.grid(True)
else:
    print("⚠️ AltMSL column not found.")

# === Plot 2: 3D Flight Path ===
if all(col in df.columns for col in ["Longitude", "Latitude", "AltMSL"]):
    fig_3d = px.line_3d(df, x="Longitude", y="Latitude", z="AltMSL", title="3D Flight Path")
    fig_3d.show()

# === Plot 3: RPM vs Time ===
if "E1 RPM" in df.columns:
    plt.figure()
    plt.plot(df["Timestamp"], df["E1 RPM"])
    plt.title("Engine RPM vs Time")
    plt.xlabel("Time")
    plt.ylabel("RPM")
    plt.grid(True)

# === Plot 4: Fuel Flow ===
if "E1 FFlow" in df.columns:
    plt.figure()
    plt.plot(df["Timestamp"], df["E1 FFlow"])
    plt.title("Fuel Flow vs Time")
    plt.xlabel("Time")
    plt.ylabel("Fuel Flow (gph)")
    plt.grid(True)

# === Plot 5: CHT1 and EGT1 ===
if all(col in df.columns for col in ["E1 CHT1", "E1 EGT1"]):
    plt.figure()
    plt.plot(df["Timestamp"], df["E1 CHT1"], label="CHT1")
    plt.plot(df["Timestamp"], df["E1 EGT1"], label="EGT1")
    plt.title("Cylinder & Exhaust Gas Temp")
    plt.xlabel("Time")
    plt.ylabel("Temp (°F)")
    plt.legend()
    plt.grid(True)

# === Plot 6: Vertical Speed vs Altitude ===
if all(col in df.columns for col in ["VSpd", "AltMSL"]):
    plt.figure()
    plt.scatter(df["VSpd"], df["AltMSL"], alpha=0.5)
    plt.title("Vertical Speed vs Altitude")
    plt.xlabel("Vertical Speed (fpm)")
    plt.ylabel("Altitude (ft)")
    plt.grid(True)

# === Plot 7: NormAc Histogram ===
if "NormAc" in df.columns:
    plt.figure()
    sns.histplot(df["NormAc"], kde=True)
    plt.title("Normal Acceleration Distribution")
    plt.xlabel("G-Force")

# === Plot 8: Pitch and Roll ===
if all(col in df.columns for col in ["Pitch", "Roll"]):
    plt.figure()
    plt.plot(df["Timestamp"], df["Pitch"], label="Pitch")
    plt.plot(df["Timestamp"], df["Roll"], label="Roll")
    plt.title("Pitch and Roll vs Time")
    plt.xlabel("Time")
    plt.ylabel("Degrees")
    plt.legend()
    plt.grid(True)

# === Plot 9: Correlation Heatmap ===
plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")

# === Plot 10: IAS (Indicated Airspeed) ===
if "IAS" in df.columns:
    plt.figure()
    plt.plot(df["Timestamp"], df["IAS"])
    plt.title("Indicated Airspeed (IAS) vs Time")
    plt.xlabel("Time")
    plt.ylabel("Airspeed (kt)")
    plt.grid(True)

# === Plot 11: Fuel Quantity Left vs Right ===
if all(col in df.columns for col in ["FQtyL", "FQtyR"]):
    plt.figure()
    plt.plot(df["Timestamp"], df["FQtyL"], label="Fuel Left")
    plt.plot(df["Timestamp"], df["FQtyR"], label="Fuel Right")
    plt.title("Fuel Tank Levels Over Time")
    plt.xlabel("Time")
    plt.ylabel("Gallons")
    plt.legend()
    plt.grid(True)

# === Show all matplotlib plots ===
plt.show()
