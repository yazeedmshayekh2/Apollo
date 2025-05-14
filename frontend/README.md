# VehicleScan AI Frontend

A modern, responsive frontend for the Vehicle Registration Card OCR system. This interface provides a user-friendly way to interact with the AI-powered OCR capabilities.

## Structure

The frontend follows a simple structure:

```
frontend/
├── public/
│   └── index.html       # Main HTML file
├── src/
│   ├── assets/
│   │   └── images/      # Icons and images
│   ├── components/      # For future component-based architecture
│   ├── styles/          # CSS styles
│   │   └── main.css     # Main stylesheet
│   └── index.js         # Main JavaScript file
└── README.md            # This file
```

## Features

- Clean, modern UI with responsive design
- Smooth animations and transitions
- Accessible interface elements
- Integrated SVG icons
- Mobile-friendly layout

## Getting Started

To view the frontend locally:

1. Navigate to the `public` directory
2. Open `index.html` in a web browser

## Integration with Backend

This frontend is designed to work with the OCR API backend. The `OCRService` class in `index.js` provides a mock implementation of the API service that would be replaced with actual API calls in production.

## Customization

- Colors and design variables can be modified in the `main.css` file
- Icons can be replaced with custom SVGs in the `assets/images` directory
- New components can be added to the `components` directory as the application grows

## Future Enhancements

- Implement a component-based architecture (React, Vue, etc.)
- Add drag-and-drop file upload functionality
- Implement real-time OCR processing status updates
- Add user authentication and history tracking
- Create a dashboard for viewing past OCR results

## License

See the main project license file for details. 