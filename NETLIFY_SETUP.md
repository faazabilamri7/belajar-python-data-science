# Setup Netlify Production Deployment

## Panduan Lengkap Deploy ke Netlify

### ✅ Step 1: Persiapan Repository

Pastikan semua changes sudah di-commit ke GitHub:

```bash
cd "d:\PORTOFOLIO PROJECT\belajar-python"
git add .
git commit -m "Setup Netlify configuration"
git push origin main
```

### ✅ Step 2: Login ke Netlify

1. Buka https://app.netlify.com
2. Klik "Sign up" atau "Log in" dengan akun GitHub Anda
3. Pilih "Sign up with GitHub"

### ✅ Step 3: Connect Repository

1. Klik "Add new site" → "Import an existing project"
2. Pilih GitHub sebagai hosting provider
3. Authorize Netlify untuk akses GitHub
4. Cari repository "belajar-python"
5. Klik "Connect & authorize"

### ✅ Step 4: Configure Build Settings

Netlify akan auto-detect konfigurasi, tapi pastikan:

**Build Settings:**

- Build command: `npm run build`
- Publish directory: `dist`
- Environment: Node 20

Klik "Deploy site"

### ✅ Step 5: Domain Configuration (Opsional)

Untuk custom domain:

1. Buka site settings
2. Klik "Domain settings"
3. Pilih "Add custom domain"
4. Masukkan domain Anda (misal: belajar-python.com)
5. Follow instruksi DNS configuration

### ✅ Step 6: Environment Variables (Jika Ada)

Jika ada secrets atau env vars:

1. Buka "Site settings"
2. Klik "Build & deploy"
3. Klik "Environment"
4. Add variables sesuai kebutuhan

### 📊 Monitoring Deployment

#### View Deployment History:

- Buka "Deployments" tab
- Lihat status build & deploy

#### View Logs:

- Klik deploy yang ingin di-check
- Scroll down untuk lihat build logs
- Debug error di sini

#### View Live Site:

- Deploy berhasil = Green checkmark
- Klik link untuk view live site

### 🔄 Automatic Deployments

**Production:**

- Setiap push ke `main` branch = auto deploy

**Preview:**

- Setiap pull request = auto preview

**Branch Deploys:**

- Setiap push ke branch lain = branch preview

### 🚀 Manual Deployment

Jika ingin manual trigger:

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod
```

### 📱 Deployment Commands

```bash
# Development build
npm run dev

# Production build (local test)
npm run build
npm run preview

# Deploy to Netlify
netlify deploy --prod

# Check status
netlify status

# View live site
netlify open:admin
```

### ⚡ Performance Tips

1. **Image Optimization:**

   - Gunakan WebP format
   - Compress images sebelum upload
   - Astro auto-optimize images

2. **Cache Strategy:**

   - Static files cached 1 tahun
   - HTML cached always fresh
   - CSS/JS cached immutable

3. **Build Optimization:**
   - Parallel builds enabled
   - Automatic minification
   - Tree-shaking active

### 🔒 Security Settings

1. **HTTPS:** Auto-enabled (Let's Encrypt)
2. **Headers:** Configured di netlify.toml
3. **CSP:** Default security headers included

### 💡 Troubleshooting

**Build gagal:**

1. Check build logs di Netlify dashboard
2. Jalankan `npm run build` locally untuk debug
3. Check Node version compatibility

**Site tidak update:**

1. Hard refresh browser (Ctrl+Shift+R)
2. Clear Netlify cache
3. Trigger rebuild

**Slow performance:**

1. Check Lighthouse scores di Netlify
2. Optimize images
3. Reduce bundle size

### 📞 Support

- Dokumentasi: https://docs.netlify.com
- Astro Guide: https://docs.astro.build/deploy/netlify
- Community: https://discord.gg/astro
