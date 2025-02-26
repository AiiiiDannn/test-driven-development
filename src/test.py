from scipy.stats import norm

hit_rate = 0.9
fa_rate = 0.1

z_hit = norm.ppf(hit_rate)  # 1.2816
z_fa = norm.ppf(fa_rate)  # -1.2816

criterion = -0.5 * (z_hit + z_fa)  # -0.5 * (1.2816 + (-1.2816)) = 0
print("z_hit:", z_hit)
print("z_fa:", z_fa)
print("criterion:", criterion)
