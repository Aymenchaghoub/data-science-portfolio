# Portfolio Website - Chaghoub Aymen

A modern, responsive React.js portfolio website showcasing projects, skills, and personal information.

## Features

- **Responsive Design**: Mobile-first approach with modern CSS Grid and Flexbox
- **React Router**: Single Page Application with smooth navigation
- **Modern UI**: Clean design with theme color #6a1b9a (purple)
- **Smooth Animations**: Fade-in scroll animations for better user experience
- **Interactive Components**: Contact form with validation and fake submission

## Pages

1. **Home**: Introduction with quick navigation cards
2. **Projects**: Showcase of 4 projects with detailed descriptions
3. **Skills**: Organized skill categories with progress indicators
4. **About**: Personal story, timeline, and values
5. **Contact**: Contact form and social links

## Technologies Used

- React 18.2.0
- React Router DOM 6.20.1
- Vite (Build Tool)
- Modern CSS (Grid, Flexbox, Custom Properties)
- Intersection Observer API (Scroll Animations)

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173`

## Project Structure

```
src/
├── components/
│   ├── Navbar.jsx & Navbar.css
│   └── Footer.jsx & Footer.css
├── pages/
│   ├── HomePage.jsx & HomePage.css
│   ├── ProjectsPage.jsx & ProjectsPage.css
│   ├── SkillsPage.jsx & SkillsPage.css
│   ├── AboutPage.jsx & AboutPage.css
│   └── ContactPage.jsx & ContactPage.css
├── utils/
│   └── scrollAnimations.js
├── App.jsx & App.css
├── index.css
└── main.jsx
```

## Features Details

### Responsive Navigation
- Fixed navbar with smooth scrolling
- Mobile hamburger menu
- Active page highlighting

### Project Showcase
- School Management App (PHP/MySQL)
- Netflix Movie Classification (Python/ML)
- Pacman Game (Python/Pygame)
- E-commerce Website (HTML/CSS/JS)

### Skills Display
- Categorized skills with color coding
- Progress indicators
- Currently learning section

### Contact Form
- Form validation
- Fake submission with alert
- Social media links

## Customization

The theme color can be easily changed by updating the CSS custom properties:
- Primary: #6a1b9a
- Secondary: #9c27b0

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This project is for portfolio purposes.
