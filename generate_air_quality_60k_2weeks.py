import random
import csv
import numpy as np
from datetime import datetime, timedelta

def generate_jakarta_air_quality_data(
    start_date_str="2025-07-01 00:00:00",
    end_date_str="2025-07-14 23:59:00",
    output_file="jakarta_air_quality_dataset.csv",
    interval_minutes=1
):
    """
    Generate realistic air quality dataset yang merepresentasikan kondisi Jakarta
    selama tanggal 1 Juli - 14 Juli dengan pengambilan data per menit.
    
    Rentang polusi berdasarkan skripsi:
    - CO (MQ7): 0-10 ppm (Sehat), 10-20 ppm (Sedang), >20 ppm (Tidak Sehat)
    - CO2 (MQ135): 400-800 ppm (Sehat), 800-2000 ppm (Sedang), >2000 ppm (Tidak Sehat)
    """
    
    print(f"Generating Jakarta air quality dataset from {start_date_str} to {end_date_str}...")
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
    
    # Definisi rentang nilai berdasarkan skripsi
    CO_RANGES = {
        'Sehat': (0, 10),      # 0-10 ppm (aman)
        'Sedang': (10, 20),    # 10-20 ppm (waspada)
        'Tidak Sehat': (20, 35) # >20 ppm (berbahaya)
    }
    
    CO2_RANGES = {
        'Sehat': (400, 800),    # 400-800 ppm (normal)
        'Sedang': (800, 2000),  # 800-2000 ppm (waspada)
        'Tidak Sehat': (2000, 3500) # >2000 ppm (bahaya)
    }
    
    # Jakarta traffic pattern factors
    # Key times in Jakarta traffic:
    WEEKDAY_RUSH_MORNING = range(6, 10)     # 6 AM - 10 AM 
    WEEKDAY_RUSH_EVENING = range(16, 20)    # 4 PM - 8 PM
    WEEKEND_BUSY_HOURS = range(10, 22)      # 10 AM - 10 PM (shopping malls, recreation)
    
    # Function to determine specific time-of-day influence
    def get_time_factor(dt):
        hour = dt.hour
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        
        # Base factor
        base = 1.0
        
        # Is weekend (Saturday=5, Sunday=6)
        is_weekend = weekday >= 5
        
        if is_weekend:
            if hour in WEEKEND_BUSY_HOURS:
                return base * random.uniform(1.1, 1.4)  # Weekend busy hours
            elif 0 <= hour < 6:
                return base * random.uniform(0.5, 0.7)  # Early morning weekend
            else:
                return base * random.uniform(0.7, 1.0)  # Other weekend hours
        else:  # Weekdays
            if hour in WEEKDAY_RUSH_MORNING:
                return base * random.uniform(1.5, 2.0)  # Morning rush hour (worst pollution)
            elif hour in WEEKDAY_RUSH_EVENING:
                return base * random.uniform(1.3, 1.8)  # Evening rush hour
            elif 10 <= hour < 16:
                return base * random.uniform(1.0, 1.3)  # Working hours
            elif 20 <= hour < 22:
                return base * random.uniform(0.8, 1.1)  # Evening, post rush
            else:
                return base * random.uniform(0.5, 0.8)  # Night/early morning
    
    # Definitions for air quality status determination
    def get_status_mq7(co_ppm):
        if co_ppm < CO_RANGES['Sedang'][0]:
            return "Sehat"
        elif co_ppm < CO_RANGES['Tidak Sehat'][0]:
            return "Sedang" 
        else:
            return "Tidak Sehat"
    
    def get_status_mq135(co2_ppm):
        if co2_ppm < CO2_RANGES['Sedang'][0]:
            return "Sehat"
        elif co2_ppm < CO2_RANGES['Tidak Sehat'][0]:
            return "Sedang"
        else:
            return "Tidak Sehat"
    
    def determine_overall_quality(co_status, co2_status):
        """
        Determines overall air quality based on CO and CO2 status
        Using the worst-case approach
        """
        status_values = {
            "Sehat": 1,
            "Sedang": 2,
            "Tidak Sehat": 3
        }
        
        max_pollution_level = max(status_values[co_status], status_values[co2_status])
        
        if max_pollution_level == 1:
            return "Baik", 1.0
        elif max_pollution_level == 2:
            return "Sedang", 2.0
        else:
            return "Tidak Sehat", 3.0
    
    # Generate random pollution patterns with some continuity
    # Use random walk to simulate continuous changes in air quality
    co_level = random.uniform(5, 10)  # Starting CO level
    co2_level = random.uniform(600, 900)  # Starting CO2 level
    
    # Create trends for different parts of the day
    # Morning trend, afternoon trend, evening trend, night trend
    trend_change_hours = [6, 10, 16, 20, 0]  # Hours when trends might change
    
    # Save weather conditions for each day to maintain consistency
    weather_conditions = {}  # key: date, value: weather factor (>1 means poor dispersion)
    
    current_time = start_date
    one_interval = timedelta(minutes=interval_minutes)
    
    # Create some random events for increased realism
    random_events = []
    current_day = start_date.date()
    while current_day <= end_date.date():
        # Each day has a 30% chance of a random pollution event
        if random.random() < 0.3:
            # Generate 1-3 events per selected day
            num_events = random.randint(1, 3)
            for _ in range(num_events):
                event_hour = random.randint(7, 20)  # Events typically happen during active hours
                event_minute = random.randint(0, 59)
                event_time = datetime.combine(current_day, datetime.min.time()) + timedelta(hours=event_hour, minutes=event_minute)
                duration_minutes = random.randint(20, 120)  # Events last 20-120 minutes
                intensity = random.uniform(1.2, 2.5)  # How much it affects pollution
                random_events.append((event_time, duration_minutes, intensity))
        
        # Set daily weather factor (affects pollution dispersion)
        weather_factor = random.choices(
            [random.uniform(0.8, 0.9),  # Good weather (good dispersion)
             random.uniform(0.9, 1.1),  # Normal weather
             random.uniform(1.1, 1.5)], # Poor weather (poor dispersion)
            weights=[0.2, 0.6, 0.2])[0]  # Probability of each condition
        weather_conditions[current_day] = weather_factor
        
        current_day += timedelta(days=1)
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["Timestamp","MQ7_CO_PPM","MQ7_Status",
                         "MQ135_CO2_PPM","MQ135_Status",
                         "Total_Pollution_PPM","Total_Pollution",
                         "Air_Quality_Status"])
        
        total_rows = 0
        
        while current_time <= end_date:
            # Get time and date factors
            time_factor = get_time_factor(current_time)
            day_factor = weather_conditions[current_time.date()]
            
            # Check if we're in any random event
            event_factor = 1.0
            for event_time, duration, intensity in random_events:
                if event_time <= current_time < (event_time + timedelta(minutes=duration)):
                    event_factor = intensity
                    break
            
            # Calculate combined factor
            combined_factor = time_factor * day_factor * event_factor
            
            # Create smoother transitions with small random walks
            co_change = random.normalvariate(0, 0.7) * combined_factor
            co2_change = random.normalvariate(0, 15) * combined_factor
            
            # Apply changes with limits to maintain realism
            co_level = max(0.5, min(35, co_level + co_change))
            co2_level = max(390, min(3500, co2_level + co2_change))
            
            # Apply time-of-day specific adjustments
            hour = current_time.hour
            if hour in WEEKDAY_RUSH_MORNING and current_time.weekday() < 5:
                # Rush hour concentrates CO more than CO2
                co_level = max(co_level, random.uniform(8, 22) * combined_factor * 0.9) 
            elif hour in WEEKDAY_RUSH_EVENING and current_time.weekday() < 5:
                # Evening rush hour
                co_level = max(co_level, random.uniform(7, 20) * combined_factor * 0.9)
            elif 0 <= hour < 5:
                # Early morning has lower levels
                co_level = min(co_level, random.uniform(2, 8))
                co2_level = min(co2_level, random.uniform(400, 700))
            
            # Get status
            mq7_status = get_status_mq7(co_level)
            mq135_status = get_status_mq135(co2_level)
            
            # Calculate total pollution
            total_ppm = co_level + co2_level
            
            # Determine overall quality
            air_quality_status, total_pollution = determine_overall_quality(mq7_status, mq135_status)
            
            # Write the data
            writer.writerow([
                current_time.strftime("%Y-%m-%d %H:%M:%S"),
                round(co_level, 2),
                mq7_status,
                round(co2_level, 2),
                mq135_status,
                round(total_ppm, 2),
                total_pollution,
                air_quality_status
            ])
            
            total_rows += 1
            current_time += one_interval
    
    print(f"Dataset successfully generated: {output_file}")
    print(f"Created {total_rows:,} rows of data")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Realistic pollution patterns incorporated for Jakarta traffic patterns")
    print("CO levels based on thesis ranges: 0-10 ppm (Sehat), 10-20 ppm (Sedang), >20 ppm (Tidak Sehat)")
    print("CO2 levels based on thesis ranges: 400-800 ppm (Sehat), 800-2000 ppm (Sedang), >2000 ppm (Tidak Sehat)")

def generate_sample_output(dataset_path, num_samples=10):
    """
    Print sample rows from the generated dataset
    """
    import pandas as pd
    
    df = pd.read_csv(dataset_path)
    print("\nSample data from generated dataset:")
    print("-" * 80)
    print(df.sample(num_samples).to_string(index=False))
    
    # Print some statistics
    print("\nDataset statistics:")
    print("-" * 80)
    print(f"Total rows: {len(df):,}")
    
    print("\nDistribution of air quality status:")
    status_counts = df['Air_Quality_Status'].value_counts()
    for status, count in status_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {status}: {count:,} ({percentage:.1f}%)")
    
    print("\nCO (MQ7) statistics:")
    print(f"  Min: {df['MQ7_CO_PPM'].min():.2f} ppm")
    print(f"  Max: {df['MQ7_CO_PPM'].max():.2f} ppm")
    print(f"  Mean: {df['MQ7_CO_PPM'].mean():.2f} ppm")
    
    print("\nCO2 (MQ135) statistics:")
    print(f"  Min: {df['MQ135_CO2_PPM'].min():.2f} ppm")
    print(f"  Max: {df['MQ135_CO2_PPM'].max():.2f} ppm")
    print(f"  Mean: {df['MQ135_CO2_PPM'].mean():.2f} ppm")
    
    # Check for data at different times of day
    df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    
    print("\nMean CO levels by hour of day (Jakarta traffic patterns):")
    hourly_co = df.groupby('hour')['MQ7_CO_PPM'].mean().reset_index()
    for _, row in hourly_co.iterrows():
        print(f"  {row['hour']:02d}:00 - {row['MQ7_CO_PPM']:.2f} ppm")

if __name__ == "__main__":
    output_file = "jakarta_air_quality_dataset.csv"
    
    # Generate dataset
    generate_jakarta_air_quality_data(
        start_date_str="2025-07-01 00:00:00",
        end_date_str="2025-07-14 23:59:00",
        output_file=output_file,
        interval_minutes=1  # Data every 1 minute
    )
    
    # Show sample output
    try:
        generate_sample_output(output_file)
    except ImportError:
        print("\nNote: Install pandas to see dataset statistics ('pip install pandas')")