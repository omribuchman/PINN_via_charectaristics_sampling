def run_buckley_leverett():
    # יחס הצמיגויות (M=2 הוא סטנדרטי לנפט/מים)
    M = 2.0
    
    # 1. פונקציית השטף (רק בשביל התיעוד, לא משתמשים בה ישירות באופיינים)
    def flux(u):
        return (u**2) / (u**2 + M * (1 - u)**2)

    # 2. הנגזרת של השטף - זו המהירות של האופיינים! a(u) = f'(u)
    def flux_derivative(x, t, u):
        # נגזרת מנה (Quotient Rule) של פונקציית באקלי-לברט
        numerator = 2 * M * u * (1 - u)
        denominator = (u**2 + M * (1 - u)**2)**2
        return numerator / denominator

    # המקדמים למערכת הקוואזי-לינארית
    def a(x, t, u): return flux_derivative(x, t, u)
    def b(x, t, u): return 1.0
    def c(x, t, u): return 0.0 # ללא מקור חיצוני

    # 3. תנאי התחלה מעניין ("מדרגה מוחלקת")
    # מדמה הזרקת מים בלחץ (u=1) לתוך מאגר מלא נפט (u=0)
    def u_init(x):
        return 0.5 * (1 - np.tanh(10 * x)) # ירידה מ-1 ל-0 סביב x=0

    # הרצה
    solver = CharacteristicSolver(a, b, c)
    x0_points = np.linspace(-1, 1, 80)
    
    print("Computing Buckley-Leverett characteristics...")
    curves = solver.solve(x0_points, u_init, t_max=0.5)
    
    solver.plot_characteristics(curves, title=f"Buckley-Leverett (M={M}) Characteristics")
    
    # בונוס: הדפסת צורת פונקציית השטף כדי להבין את הפיזיקה
    u_vals = np.linspace(0, 1, 100)
    f_vals = flux(u_vals)
    plt.figure(figsize=(6,4))
    plt.plot(u_vals, f_vals)
    plt.title("Buckley-Leverett Flux Function (Non-Convex)")
    plt.xlabel("u (Saturation)")
    plt.ylabel("f(u) (Flux)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_buckley_leverett()