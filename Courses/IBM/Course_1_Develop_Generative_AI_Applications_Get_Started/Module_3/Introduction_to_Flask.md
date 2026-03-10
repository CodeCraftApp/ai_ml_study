# Introduction to Flask

## Learning Objectives

After completing this introduction to Flask, you will be able to:
- Define the Flask web framework and describe its main features
- Explain how to install Flask on your machines
- Describe the main dependencies of Flask
- Explain the main differences between Flask and Django

## What is Flask?

Flask is a **micro framework** that can create web applications. Unlike some other larger frameworks, Flask is not opinionated and does not bind the user to a specific set of tools.

### Key Points

- **Framework Type**: Light-weight, flexible micro framework
- **Python Dependency**: Flask 2.2.2 requires a minimum Python version of 3.7
- **History**: Created by Armin Ronacher in 2004 as an April Fool's joke, it quickly gained popularity for its ease of use and extensibility
- **Extensibility**: Flask provides minimal dependencies needed to create a web application, but is extensible—many community extensions add additional features

## Main Features of Flask

### Core Features

**Web Server and Debugging**
- Flask has a built-in web server that runs applications in development mode
- Comes with a debugger to help debug applications
- The debugger shows interactive traceback and stack trace in the browser

**Logging and Testing**
- Uses standard Python logging for application logs
- You can use the same logger to log custom messages about your application
- Provides a way to test different parts of your application
- The testing feature enables developers to follow a test-driven approach
- You can use frameworks like Pytest and coverage to ensure your code works as desired

**Request and Response Handling**
- Developers can access the request and response objects to pull arguments and customize responses

### Additional Features

**Static Assets**
- The framework supports static assets like CSS files, JavaScript files, and images
- Flask provides tags to load static files in templates

**Dynamic Pages**
- Develop dynamic pages using the Jinja templating framework
- Dynamic pages can display information that may change for each request
- Can check if the user is logged in and display content accordingly

**Routing and URLs**
- Flask provides routing and supports dynamic URLs
- Extremely useful for RESTful services
- You can create routes for different HTTP methods
- Provides redirection in your application

**Error Handling and Sessions**
- Write global error handlers in Flask that work on the application level
- Supports user session management

## Popular Community Extensions

Flask can be extended with popular community extensions:

### Database and ORM

**Flask-SQLAlchemy**
- Adds support for SQLAlchemy ORM to Flask
- Gives developers a way to work with database objects in Python

### Data and Files

**Flask-Mail**
- Provides the ability to set up an SMTP mail server

**Flask-Uploads**
- Allows you to add customized file uploading to your application

### Admin and Utilities

**Flask-Admin**
- Lets you add admin interfaces to Flask applications easily

**Flask-CORS**
- Allows your application to handle Cross-Origin Resource Sharing (CORS)
- Makes cross-origin JavaScript requests possible

**Flask-Migrate**
- Adds database migrations to SQLAlchemy ORM

**Flask-User**
- Adds user authentication, authorization, and other user management activities

**Marshmallow**
- Adds extensive object serialization and deserialization support to your code

**Celery**
- A powerful task queue that can be used for:
  - Simple background tasks
  - Complex multi-storage programs
  - Job scheduling

## Installing Flask

### Requirements

- Flask is available on **pip**, the Python package manager
- Pip is available in the lab environment
- If installing on your machines, it is recommended that you first create a virtual environment using the `venv` or `binvenv` module

### Installation Steps

1. Create a virtual environment (recommended)
2. Install Flask version 2.2.2

### Version Pinning

It is recommended to pin the version number of the dependencies in your application. This ensures:
- The application can be reproduced from scratch in different environments (development, staging, production)
- New issues and bugs are not introduced by mistake when packages are updated automatically

### Checking Dependencies

You can use the `pip freeze` command in the virtual environment to see all installed packages, including built-in Flask dependencies.

## Built-in Dependencies

Flask comes with several built-in dependencies that enable various features:

**Werkzeug**
- Implements WSGI (Web Server Gateway Interface)
- The standard Python interface between applications and servers

**Jinja**
- A template language that renders the pages in your application

**MarkupSafe**
- Comes with Jinja
- Escapes untrusted input when rendering templates to avoid injection attacks

**ItsDangerous**
- Used to assign data securely
- Helps determine if data has been tampered with
- Used to protect Flask session cookies

**Click**
- A framework for writing command-line applications
- Provides the Flask command
- Allows adding custom management commands

## Flask vs. Django

| Aspect | Flask | Django |
|--------|-------|--------|
| **Framework Type** | Micro framework (light-weight) | Full-stack framework |
| **Dependencies** | Minimal - only basic dependencies needed | Includes everything for a full-stack application |
| **Extensibility** | Flexible - add and remove pieces in a plug-and-play manner | Opinionated - makes decisions for developers |
| **Learning Curve** | Lighter, more minimal | Steeper, more comprehensive |
| **Use Case** | Best for custom, modular applications | Best for rapid development of complete applications |

## Summary

Flask is a micro framework that ships with minimal dependencies but powerful features including:
- Debugging servers
- Routing
- Templates
- Error handling
- Session management

Flask can be extended using community extensions to add additional functionality, and it can be installed as a Python package via pip. Compared to Django, Flask provides more flexibility but requires more decision-making from the developer about which tools and extensions to use.
