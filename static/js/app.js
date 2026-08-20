document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    document.querySelectorAll('.alert').forEach(el => {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 300);
    });
  }, 5000);
});