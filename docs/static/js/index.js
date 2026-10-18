// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
        button.classList.add('copied');
        copyText.textContent = 'Kopyalandı';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Kopyala';
        }, 2000);
    });
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function() {
    document.querySelector('.scroll-to-top').classList.toggle('visible', window.pageYOffset > 300);
});

$(document).ready(function() {
    bulmaCarousel.attach('.carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    });
});
