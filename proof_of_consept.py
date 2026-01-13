import numpy as np
import matplotlib.pyplot as plt

def generate_comparison_plot():
    # 1. הגדרות הבעיה (Burgers Equation)
    # u_t + u*u_x = nu * u_xx
    # תנאי התחלה שיוצר הלם: u(x,0) = -sin(pi*x)
    
    N_start = 50      # מספר נקודות התחלתיות על ציר ה-x
    T_max = 0.8       # זמן סופי (ההלם נוצר סביב t=0.32)
    nu = 0.01/np.pi   # צמיגות (רק לצורך חישוב הרעש הסטוכסטי)
    
    # ---------------------------------------------------------
    # שיטה א': דגימה אקראית רגילה (Standard PINN Sampling)
    # ---------------------------------------------------------
    num_samples = N_start * 20 # כדי שיהיה הוגן, נשתמש באותה כמות נקודות סה"כ
    
    x_uniform = np.random.uniform(-1, 1, num_samples)
    t_uniform = np.random.uniform(0, T_max, num_samples)
    
    # ---------------------------------------------------------
    # שיטה ב': השיטה שלך (Characteristic-Based Sampling)
    # ---------------------------------------------------------
    
    # 1. מגדירים נקודות בזמן t=0
    x0 = np.linspace(-1, 1, N_start)
    u0 = -np.sin(np.pi * x0) # המהירות ההתחלתית
    
    x_char = []
    t_char = []
    
    # 2. מריצים את הזמן קדימה
    # במקום להגריל זמן אקראי, אנחנו עוקבים אחרי המסלולים בזמנים בדידים
    time_steps = np.linspace(0, T_max, 20)
    
    for t in time_steps:
        # א. החלק ההיפרבולי (הדטרמיניסטי): x = x0 + u0 * t
        # המידע נע עם המהירות ההתחלתית
        x_deterministic = x0 + u0 * t
        
        # ב. החלק הפרבולי (הרעש): מדמה את הצמיגות
        # הסטייה היא פרופורציונלית לשורש הזמן והצמיגות (Brownian motion)
        diffusion_noise = np.random.normal(0, np.sqrt(2 * nu * t + 1e-10), size=x0.shape)
        
        # המיקום הסופי
        x_current = x_deterministic + diffusion_noise
        
        # שמירה לרשימה (מסננים נקודות שברחו מהתחום)
        mask = (x_current >= -1) & (x_current <= 1)
        x_char.extend(x_current[mask])
        t_char.extend([t] * np.sum(mask))

    # ---------------------------------------------------------
    # ויזואליזציה והשוואה
    # ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # גרף שמאל: רגיל
    ax[0].scatter(x_uniform, t_uniform, s=10, c='gray', alpha=0.5, label='Collocation Points')
    ax[0].set_title('Standard Uniform Sampling\n(Blind to Physics)', fontsize=14)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(0, T_max)
    ax[0].grid(True, alpha=0.3)
    
    # גרף ימין: השיטה שלך
    # אני מוסיף צבע לפי צפיפות או סתם כדי להבליט את ההתכנסות
    ax[1].scatter(x_char, t_char, s=10, c='blue', alpha=0.6, label='Characteristic Points')
    
    # ציור קו ההלם המשוער (רק בשביל העין)
    ax[1].axvline(0, color='red', linestyle='--', alpha=0.3, lw=2)
    ax[1].text(0.1, 0.7, 'Shock Formation\nRegion', color='red', fontsize=10)
    
    ax[1].set_title('Your Method: Characteristic Sampling\n(Physics-Aware)', fontsize=14, fontweight='bold')
    ax[1].set_xlabel('x')
    ax[1].set_xlim(-1, 1)
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_comparison_plot()