import numpy as np
import matplotlib.pyplot as plt

def generate_characteristics():
    # הגדרות כלליות
    t_max = 1.5
    x_max = 2.0
    
    # --- 1. הגדרת תנאי ההתחלה (IC) ---
    # נניח שבזמן t=0, הערך הוא 0.5 לאורך כל התחום
    n_ic = 15
    x_start_ic = np.linspace(0, x_max, n_ic)
    t_start_ic = np.zeros(n_ic)          # כולם מתחילים בזמן 0
    u_start_ic = np.ones(n_ic) * 0.5     # ערך התחלתי: 0.5
    
    # --- 2. הגדרת תנאי השפה (BC) ---
    # נניח שבקצה x=0, אנחנו מזרימים פנימה ערך חזק יותר של 1.2
    n_bc = 15
    t_start_bc = np.linspace(0, t_max, n_bc)
    x_start_bc = np.zeros(n_bc)          # כולם מתחילים במיקום 0
    u_start_bc = np.ones(n_bc) * 1.2     # ערך שפה: 1.2

    # סינון: ניקח קווים מהשפה רק אם הזרימה נכנסת (u > 0)
    # במקרה שלנו 1.2 > 0 ולכן כולם נלקחים, אבל זה קריטי למקרים מורכבים
    mask_inflow = u_start_bc > 0
    t_start_bc = t_start_bc[mask_inflow]
    x_start_bc = x_start_bc[mask_inflow]
    u_start_bc = u_start_bc[mask_inflow]

    # --- 3. איחוד הרשימות ---
    # כאן אנחנו מאחדים את הנקודות לרשימה אחת גדולה לעיבוד
    # שיטת החישוב מעכשיו זהה לחלוטין לשתי הקבוצות
    x_starts = np.concatenate([x_start_ic, x_start_bc])
    t_starts = np.concatenate([t_start_ic, t_start_bc])
    u_vals   = np.concatenate([u_start_ic, u_start_bc])
    
    # --- 4. חישוב ושרטוט הקווים ---
    plt.figure(figsize=(10, 6))
    
    # זמני עתיד שנרצה לחשב עבורם
    t_future = np.linspace(0, t_max, 100)
    
    for i in range(len(x_starts)):
        u = u_vals[i]
        t0 = t_starts[i]
        x0 = x_starts[i]
        
        # חישוב המסלול: x(t) = x0 + u * (t - t0)
        # שים לב: אנחנו מחשבים רק לזמנים שגדולים מ-t0
        valid_times = t_future[t_future >= t0]
        if len(valid_times) == 0: continue
            
        x_path = x0 + u * (valid_times - t0)
        
        # צביעה שונה כדי להמחיש את המקור
        if t0 == 0: 
            color = 'blue' # מקור: תנאי התחלה
            label = 'From IC (t=0)' if i == 0 else ""
        else: 
            color = 'red'  # מקור: תנאי שפה
            label = 'From BC (x=0)' if i == len(x_start_ic) else ""
            
        plt.plot(x_path, valid_times, color=color, alpha=0.7, label=label, linewidth=2)

    # --- עיצוב הגרף ---
    plt.xlabel('Space (x)')
    plt.ylabel('Time (t)')
    plt.title('Method of Characteristics: IC vs BC Origins')
    plt.xlim(0, x_max)
    plt.ylim(0, t_max)
    plt.axhline(0, color='black', linewidth=1) # ציר זמן
    plt.axvline(0, color='black', linewidth=1) # ציר מקום
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

generate_characteristics()