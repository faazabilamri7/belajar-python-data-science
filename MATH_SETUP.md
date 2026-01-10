# ✅ Math Formula Setup - KaTeX Configuration

## 📋 Setup Selesai!

Kami sudah mengkonfigurasi KaTeX untuk render formula matematika di Astro project Anda.

## 🔧 Apa yang Sudah Diinstall

### Dependencies yang ditambahkan:

- ✅ `remark-math` - Plugin untuk parse math syntax
- ✅ `rehype-katex` - Plugin untuk render ke KaTeX
- ✅ `katex` - Math rendering library

## 📝 Cara Menggunakan

### Inline Formula (dalam paragraf)

```markdown
Rumus Pythagoras: $a^2 + b^2 = c^2$

atau dengan escape: \$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\$
```

### Block Formula (formula berdiri sendiri)

```markdown
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$
```

### Contoh Syntax Math yang Valid

#### 1. Basic Operations

```markdown
$$a + b = c$$
$$x - y = z$$
$$a \times b = c$$
$$\frac{a}{b}$$
```

#### 2. Superscript & Subscript

```markdown
$$x^2 + y^2$$
$$a_1, a_2, a_3$$
$$e^{i\pi} + 1 = 0$$
```

#### 3. Fractions & Roots

```markdown
$$\frac{numerator}{denominator}$$
$$\sqrt{x}$$
$$\sqrt[3]{x}$$
```

#### 4. Summation & Integration

```markdown
$$\sum_{i=1}^{n} x_i$$
$$\int_{0}^{\infty} e^{-x^2} dx$$
```

#### 5. Greek Letters

```markdown
$$\alpha, \beta, \gamma, \delta$$
$$\mu, \sigma, \lambda$$
$$\pi, \Sigma, \Omega$$
```

## ✨ Contoh di Files Anda

Di `pertemuan-5/statistik-dasar.md`, Anda sudah ada formula seperti:

```markdown
$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

$$\sigma^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}$$

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
```

Semua formula ini sekarang akan tampil dengan proper mathematical notation!

## 🎨 Styling (Optional)

KaTeX menggunakan CSS default yang clean dan profesional. Jika ingin customize:

1. **Ukuran font**: Buka DevTools → inspect formula → edit CSS
2. **Warna**: Tambahkan custom CSS di `src/styles/` jika dibuat
3. **Font**: KaTeX sudah include font math yang optimal

## 📱 Kompatibilitas

- ✅ Desktop browsers (Chrome, Firefox, Safari, Edge)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)
- ✅ Dark mode (KaTeX adjust otomatis)
- ✅ Print (formula print dengan sempurna)

## 🚀 Deploy ke Netlify

Semua setup sudah kompatibel dengan Netlify. Build command tetap:

```bash
npm run build
```

KaTeX CSS sudah included via CDN, jadi tidak ada masalah cache atau file size.

## ⚠️ Troubleshooting

### Formula tidak tampil?

- Pastikan spacing benar: `$$formula$$` (bukan `$$ formula$$`)
- Check console untuk error messages
- Clear browser cache (Ctrl+Shift+Delete)

### Formula tampil tapi styling jelek?

- Clear CSS cache di browser
- Refresh halaman (Ctrl+F5)
- Cek internet connection untuk CDN

### Build error?

- Delete `node_modules` & `.astro`: `rm -rf node_modules .astro && npm install`
- Run build lagi: `npm run build`

## 📚 Resources

- [KaTeX Documentation](https://katex.org/docs/support_table.html)
- [Remark Math Plugin](https://github.com/remarkjs/remark-math)
- [Astro Markdown Config](https://docs.astro.build/en/guides/markdown-content/)

---

**Setup Date**: 10 January 2026  
**Status**: ✅ Ready for Production
