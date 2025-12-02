import dash
from dash import dcc, html, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import requests
import json
from datetime import datetime
import base64

# ============= CONFIGURATION =============
API_BASE_URL = "http://localhost:8000"

# ============= DASH APP =============
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Multimodal RAG System"
)

# ============= CUSTOM CSS - ULTRA CLEAN WITH DARK MODE =============
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                overflow: hidden;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            
            /* Light Mode (Default) */
            body {
                background: #ffffff;
                color: #000000;
            }
            
            /* Dark Mode */
            body.dark-mode {
                background: #0a0a0a;
                color: #ffffff;
            }
            
            /* Layout */
            .app-container {
                display: flex;
                height: 100vh;
                width: 100%;
            }
            
            /* Sidebar */
            .sidebar {
                width: 200px;
                border-right: 1px solid;
                display: flex;
                flex-direction: column;
                transition: background-color 0.3s ease, border-color 0.3s ease;
            }
            
            body .sidebar {
                background: #fafafa;
                border-right-color: #e5e5e5;
            }
            
            body.dark-mode .sidebar {
                background: #0f0f0f;
                border-right-color: #1a1a1a;
            }
            
            .sidebar-header {
                padding: 32px 24px;
                border-bottom: 1px solid;
                transition: border-color 0.3s ease;
            }
            
            body .sidebar-header {
                border-bottom-color: #e5e5e5;
            }
            
            body.dark-mode .sidebar-header {
                border-bottom-color: #1a1a1a;
            }
            
            .sidebar-logo {
                font-size: 18px;
                font-weight: 700;
                letter-spacing: -0.5px;
                transition: color 0.3s ease;
            }
            
            body .sidebar-logo {
                color: #000000;
            }
            
            body.dark-mode .sidebar-logo {
                color: #ffffff;
            }
            
            .sidebar-menu {
                flex: 1;
                padding: 24px 0;
            }
            
            .menu-item {
                padding: 12px 24px;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 14px;
                font-weight: 500;
                border-left: 2px solid transparent;
            }
            
            body .menu-item {
                color: #737373;
            }
            
            body.dark-mode .menu-item {
                color: #737373;
            }
            
            body .menu-item:hover {
                background: #f5f5f5;
                color: #000000;
            }
            
            body.dark-mode .menu-item:hover {
                background: #1a1a1a;
                color: #ffffff;
            }
            
            body .menu-item.active {
                background: #f5f5f5;
                color: #000000;
                border-left-color: #000000;
                font-weight: 600;
            }
            
            body.dark-mode .menu-item.active {
                background: #1a1a1a;
                color: #ffffff;
                border-left-color: #ffffff;
                font-weight: 600;
            }
            
            .sidebar-footer {
                padding: 24px;
                border-top: 1px solid;
                transition: border-color 0.3s ease;
            }
            
            body .sidebar-footer {
                border-top-color: #e5e5e5;
            }
            
            body.dark-mode .sidebar-footer {
                border-top-color: #1a1a1a;
            }
            
            .stats-mini {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .stat-mini {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 12px;
            }
            
            .stat-mini-label {
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .stat-mini-label {
                color: #737373;
            }
            
            body.dark-mode .stat-mini-label {
                color: #737373;
            }
            
            .stat-mini-value {
                font-weight: 700;
                transition: color 0.3s ease;
            }
            
            body .stat-mini-value {
                color: #000000;
            }
            
            body.dark-mode .stat-mini-value {
                color: #ffffff;
            }
            
            /* Dark Mode Toggle */
            .dark-mode-toggle {
                margin-top: 16px;
                padding: 10px 0;
                border-top: 1px solid;
                transition: border-color 0.3s ease;
            }
            
            body .dark-mode-toggle {
                border-top-color: #e5e5e5;
            }
            
            body.dark-mode .dark-mode-toggle {
                border-top-color: #1a1a1a;
            }
            
            .toggle-button {
                width: 100%;
                padding: 8px 12px;
                border-radius: 6px;
                border: 1px solid;
                cursor: pointer;
                font-size: 12px;
                font-weight: 500;
                transition: all 0.2s ease;
                text-align: center;
            }
            
            body .toggle-button {
                background: #ffffff;
                border-color: #e5e5e5;
                color: #000000;
            }
            
            body.dark-mode .toggle-button {
                background: #1a1a1a;
                border-color: #262626;
                color: #ffffff;
            }
            
            body .toggle-button:hover {
                background: #f5f5f5;
                border-color: #d4d4d4;
            }
            
            body.dark-mode .toggle-button:hover {
                background: #262626;
                border-color: #333333;
            }
            
            /* Main Content */
            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                transition: background-color 0.3s ease;
            }
            
            body .main-content {
                background: #ffffff;
            }
            
            body.dark-mode .main-content {
                background: #0a0a0a;
            }
            
            /* Header Bar */
            .header-bar {
                height: 64px;
                border-bottom: 1px solid;
                display: flex;
                align-items: center;
                padding: 0 32px;
                justify-content: space-between;
                transition: background-color 0.3s ease, border-color 0.3s ease;
            }
            
            body .header-bar {
                background: #ffffff;
                border-bottom-color: #e5e5e5;
            }
            
            body.dark-mode .header-bar {
                background: #0a0a0a;
                border-bottom-color: #1a1a1a;
            }
            
            .header-title {
                font-size: 16px;
                font-weight: 600;
                letter-spacing: -0.3px;
                transition: color 0.3s ease;
            }
            
            body .header-title {
                color: #000000;
            }
            
            body.dark-mode .header-title {
                color: #ffffff;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
                font-weight: 500;
                color: #22c55e;
            }
            
            .status-indicator.disconnected {
                color: #ef4444;
            }
            
            .status-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: currentColor;
            }
            
            /* Content Area */
            .content-area {
                flex: 1;
                overflow: hidden;
                display: flex;
            }
            
            /* Chat Container */
            .chat-main {
                flex: 1;
                display: flex;
                flex-direction: column;
                transition: background-color 0.3s ease;
            }
            
            body .chat-main {
                background: #ffffff;
            }
            
            body.dark-mode .chat-main {
                background: #0a0a0a;
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 32px;
            }
            
            .message {
                margin-bottom: 24px;
            }
            
            .message-user {
                display: flex;
                justify-content: flex-end;
            }
            
            .message-assistant {
                display: flex;
                justify-content: flex-start;
            }
            
            .message-wrapper {
                max-width: 70%;
                display: flex;
                flex-direction: column;
            }
            
            .message-bubble {
                padding: 16px 20px;
                border-radius: 12px;
                word-wrap: break-word;
                line-height: 1.6;
                font-size: 14px;
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            
            body .message-user .message-bubble {
                background: #000000;
                color: #ffffff;
            }
            
            body.dark-mode .message-user .message-bubble {
                background: #ffffff;
                color: #000000;
            }
            
            body .message-assistant .message-bubble {
                background: #f5f5f5;
                color: #000000;
            }
            
            body.dark-mode .message-assistant .message-bubble {
                background: #1a1a1a;
                color: #ffffff;
            }
            
            .message-meta {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-top: 8px;
                font-size: 11px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .message-meta {
                color: #a3a3a3;
            }
            
            body.dark-mode .message-meta {
                color: #525252;
            }
            
            .message-user .message-meta {
                justify-content: flex-end;
            }
            
            /* Sources Section */
            .sources-container {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid;
                transition: border-color 0.3s ease;
            }
            
            body .sources-container {
                border-top-color: #e5e5e5;
            }
            
            body.dark-mode .sources-container {
                border-top-color: #1a1a1a;
            }
            
            .sources-header {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 12px;
                transition: color 0.3s ease;
            }
            
            body .sources-header {
                color: #737373;
            }
            
            body.dark-mode .sources-header {
                color: #737373;
            }
            
            .source-item {
                padding: 12px 16px;
                border-radius: 8px;
                margin-bottom: 8px;
                transition: all 0.2s ease;
                cursor: pointer;
                border: 1px solid transparent;
            }
            
            body .source-item {
                background: #fafafa;
            }
            
            body.dark-mode .source-item {
                background: #0f0f0f;
            }
            
            body .source-item:hover {
                background: #f5f5f5;
                border-color: #e5e5e5;
            }
            
            body.dark-mode .source-item:hover {
                background: #1a1a1a;
                border-color: #262626;
            }
            
            .source-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 6px;
            }
            
            .source-name {
                font-size: 12px;
                font-weight: 600;
                transition: color 0.3s ease;
            }
            
            body .source-name {
                color: #000000;
            }
            
            body.dark-mode .source-name {
                color: #ffffff;
            }
            
            .source-type {
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: color 0.3s ease;
            }
            
            body .source-type {
                color: #737373;
            }
            
            body.dark-mode .source-type {
                color: #737373;
            }
            
            .source-meta {
                font-size: 11px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .source-meta {
                color: #a3a3a3;
            }
            
            body.dark-mode .source-meta {
                color: #525252;
            }
            
            /* Images Grid */
            .images-section {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid;
                transition: border-color 0.3s ease;
            }
            
            body .images-section {
                border-top-color: #e5e5e5;
            }
            
            body.dark-mode .images-section {
                border-top-color: #1a1a1a;
            }
            
            .images-header {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 12px;
                transition: color 0.3s ease;
            }
            
            body .images-header {
                color: #737373;
            }
            
            body.dark-mode .images-header {
                color: #737373;
            }
            
            .images-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 12px;
            }
            
            .image-card {
                border-radius: 8px;
                overflow: hidden;
                transition: all 0.2s ease;
                cursor: pointer;
                border: 1px solid;
            }
            
            body .image-card {
                background: #fafafa;
                border-color: #e5e5e5;
            }
            
            body.dark-mode .image-card {
                background: #0f0f0f;
                border-color: #1a1a1a;
            }
            
            body .image-card:hover {
                border-color: #d4d4d4;
                transform: translateY(-2px);
            }
            
            body.dark-mode .image-card:hover {
                border-color: #262626;
                transform: translateY(-2px);
            }
            
            .image-wrapper {
                position: relative;
                width: 100%;
                padding-top: 75%;
                transition: background-color 0.3s ease;
            }
            
            body .image-wrapper {
                background: #ffffff;
            }
            
            body.dark-mode .image-wrapper {
                background: #0a0a0a;
            }
            
            .image-wrapper img {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            .image-info {
                padding: 10px 12px;
            }
            
            .image-label {
                font-size: 11px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .image-label {
                color: #737373;
            }
            
            body.dark-mode .image-label {
                color: #737373;
            }
            
            /* Modal */
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                padding: 40px;
            }
            
            .modal-overlay.active {
                display: flex;
            }
            
            .modal-content {
                border-radius: 12px;
                padding: 0;
                max-width: 900px;
                width: 100%;
                max-height: 90vh;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                transition: background-color 0.3s ease;
            }
            
            body .modal-content {
                background: #ffffff;
            }
            
            body.dark-mode .modal-content {
                background: #0a0a0a;
            }
            
            .modal-header {
                padding: 24px 28px;
                border-bottom: 1px solid;
                display: flex;
                align-items: center;
                justify-content: space-between;
                transition: border-color 0.3s ease;
            }
            
            body .modal-header {
                border-bottom-color: #e5e5e5;
            }
            
            body.dark-mode .modal-header {
                border-bottom-color: #1a1a1a;
            }
            
            .modal-title {
                font-size: 16px;
                font-weight: 600;
                transition: color 0.3s ease;
            }
            
            body .modal-title {
                color: #000000;
            }
            
            body.dark-mode .modal-title {
                color: #ffffff;
            }
            
            .modal-close {
                width: 32px;
                height: 32px;
                border-radius: 6px;
                border: 1px solid;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                transition: all 0.2s ease;
            }
            
            body .modal-close {
                background: #f5f5f5;
                border-color: #e5e5e5;
                color: #737373;
            }
            
            body.dark-mode .modal-close {
                background: #1a1a1a;
                border-color: #262626;
                color: #737373;
            }
            
            body .modal-close:hover {
                background: #e5e5e5;
                color: #000000;
            }
            
            body.dark-mode .modal-close:hover {
                background: #262626;
                color: #ffffff;
            }
            
            .modal-body {
                padding: 24px 28px;
                overflow-y: auto;
                flex: 1;
            }
            
            .modal-section {
                margin-bottom: 24px;
            }
            
            .modal-section:last-child {
                margin-bottom: 0;
            }
            
            .modal-section-title {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 12px;
                transition: color 0.3s ease;
            }
            
            body .modal-section-title {
                color: #737373;
            }
            
            body.dark-mode .modal-section-title {
                color: #737373;
            }
            
            .content-preview-card {
                border: 1px solid;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
                transition: all 0.2s ease;
            }
            
            body .content-preview-card {
                background: #fafafa;
                border-color: #e5e5e5;
            }
            
            body.dark-mode .content-preview-card {
                background: #0f0f0f;
                border-color: #1a1a1a;
            }
            
            .content-preview-header {
                margin-bottom: 10px;
            }
            
            .content-preview-title {
                font-weight: 600;
                font-size: 13px;
                margin-bottom: 4px;
                transition: color 0.3s ease;
            }
            
            body .content-preview-title {
                color: #000000;
            }
            
            body.dark-mode .content-preview-title {
                color: #ffffff;
            }
            
            .content-preview-meta {
                font-size: 11px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .content-preview-meta {
                color: #a3a3a3;
            }
            
            body.dark-mode .content-preview-meta {
                color: #525252;
            }
            
            .content-preview-text {
                font-size: 12px;
                line-height: 1.6;
                font-family: 'Courier New', monospace;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid;
                max-height: 120px;
                overflow-y: auto;
                transition: all 0.3s ease;
            }
            
            body .content-preview-text {
                color: #525252;
                background: #ffffff;
                border-color: #e5e5e5;
            }
            
            body.dark-mode .content-preview-text {
                color: #a3a3a3;
                background: #0a0a0a;
                border-color: #1a1a1a;
            }
            
            .content-preview-image {
                margin-top: 12px;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid;
                transition: border-color 0.3s ease;
            }
            
            body .content-preview-image {
                border-color: #e5e5e5;
            }
            
            body.dark-mode .content-preview-image {
                border-color: #1a1a1a;
            }
            
            .content-preview-image img {
                width: 100%;
                display: block;
                max-height: 300px;
                object-fit: contain;
                transition: background-color 0.3s ease;
            }
            
            body .content-preview-image img {
                background: #ffffff;
            }
            
            body.dark-mode .content-preview-image img {
                background: #0a0a0a;
            }
            
            /* Input Area */
            .chat-input-container {
                padding: 24px 32px;
                border-top: 1px solid;
                transition: background-color 0.3s ease, border-color 0.3s ease;
            }
            
            body .chat-input-container {
                background: #ffffff;
                border-top-color: #e5e5e5;
            }
            
            body.dark-mode .chat-input-container {
                background: #0a0a0a;
                border-top-color: #1a1a1a;
            }
            
            .input-wrapper {
                display: flex;
                gap: 12px;
                align-items: center;
            }
            
            .chat-input {
                flex: 1;
                border: 1px solid;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                resize: none;
                transition: all 0.2s ease;
                font-family: 'Inter', sans-serif;
                line-height: 1.5;
            }
            
            body .chat-input {
                background: #ffffff;
                border-color: #e5e5e5;
                color: #000000;
            }
            
            body.dark-mode .chat-input {
                background: #0a0a0a;
                border-color: #1a1a1a;
                color: #ffffff;
            }
            
            body .chat-input::placeholder {
                color: #a3a3a3;
            }
            
            body.dark-mode .chat-input::placeholder {
                color: #525252;
            }
            
            body .chat-input:focus {
                outline: none;
                border-color: #000000;
            }
            
            body.dark-mode .chat-input:focus {
                outline: none;
                border-color: #ffffff;
            }
            
            .send-button {
                width: 48px;
                height: 48px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s ease;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            body .send-button {
                background: #000000;
                color: #ffffff;
            }
            
            body.dark-mode .send-button {
                background: #ffffff;
                color: #000000;
            }
            
            body .send-button:hover {
                background: #262626;
            }
            
            body.dark-mode .send-button:hover {
                background: #e5e5e5;
            }
            
            body .send-button:disabled {
                background: #e5e5e5;
                color: #a3a3a3;
                cursor: not-allowed;
            }
            
            body.dark-mode .send-button:disabled {
                background: #1a1a1a;
                color: #525252;
                cursor: not-allowed;
            }
            
            /* Settings Panel */
            .settings-panel {
                width: 280px;
                border-left: 1px solid;
                padding: 24px;
                overflow-y: auto;
                transition: background-color 0.3s ease, border-color 0.3s ease;
            }
            
            body .settings-panel {
                background: #fafafa;
                border-left-color: #e5e5e5;
            }
            
            body.dark-mode .settings-panel {
                background: #0f0f0f;
                border-left-color: #1a1a1a;
            }
            
            .settings-title {
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 24px;
                transition: color 0.3s ease;
            }
            
            body .settings-title {
                color: #000000;
            }
            
            body.dark-mode .settings-title {
                color: #ffffff;
            }
            
            .settings-group {
                margin-bottom: 20px;
            }
            
            .settings-label {
                font-size: 11px;
                font-weight: 600;
                margin-bottom: 8px;
                display: block;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: color 0.3s ease;
            }
            
            body .settings-label {
                color: #737373;
            }
            
            body.dark-mode .settings-label {
                color: #737373;
            }
            
            /* Upload Area */
            .upload-container {
                padding: 48px;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }
            
            .upload-area {
                border: 2px dashed;
                border-radius: 12px;
                padding: 80px 60px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                max-width: 600px;
                width: 100%;
            }
            
            body .upload-area {
                border-color: #e5e5e5;
                background: #fafafa;
            }
            
            body.dark-mode .upload-area {
                border-color: #1a1a1a;
                background: #0f0f0f;
            }
            
            body .upload-area:hover {
                border-color: #000000;
                background: #f5f5f5;
            }
            
            body.dark-mode .upload-area:hover {
                border-color: #ffffff;
                background: #1a1a1a;
            }
            
            .upload-title {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 8px;
                transition: color 0.3s ease;
            }
            
            body .upload-title {
                color: #000000;
            }
            
            body.dark-mode .upload-title {
                color: #ffffff;
            }
            
            .upload-subtitle {
                font-size: 14px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .upload-subtitle {
                color: #737373;
            }
            
            body.dark-mode .upload-subtitle {
                color: #737373;
            }
            
            /* Documents */
            .documents-container {
                padding: 48px;
                overflow-y: auto;
            }
            
            .documents-header {
                margin-bottom: 32px;
            }
            
            .documents-title {
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 8px;
                transition: color 0.3s ease;
            }
            
            body .documents-title {
                color: #000000;
            }
            
            body.dark-mode .documents-title {
                color: #ffffff;
            }
            
            .documents-subtitle {
                font-size: 14px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .documents-subtitle {
                color: #737373;
            }
            
            body.dark-mode .documents-subtitle {
                color: #737373;
            }
            
            .documents-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 16px;
            }
            
            .document-card {
                border: 1px solid;
                padding: 20px;
                border-radius: 8px;
                transition: all 0.2s ease;
                cursor: pointer;
            }
            
            body .document-card {
                background: #fafafa;
                border-color: #e5e5e5;
            }
            
            body.dark-mode .document-card {
                background: #0f0f0f;
                border-color: #1a1a1a;
            }
            
            body .document-card:hover {
                background: #f5f5f5;
                border-color: #d4d4d4;
            }
            
            body.dark-mode .document-card:hover {
                background: #1a1a1a;
                border-color: #262626;
            }
            
            .document-name {
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 8px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                transition: color 0.3s ease;
            }
            
            body .document-name {
                color: #000000;
            }
            
            body.dark-mode .document-name {
                color: #ffffff;
            }
            
            .document-meta {
                font-size: 12px;
                display: flex;
                gap: 8px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .document-meta {
                color: #737373;
            }
            
            body.dark-mode .document-meta {
                color: #737373;
            }
            
            /* Badge */
            .badge {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }
            
            body .badge-success {
                background: #dcfce7;
                color: #166534;
            }
            
            body.dark-mode .badge-success {
                background: #14532d;
                color: #86efac;
            }
            
            /* Empty State */
            .empty-state {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                padding: 60px;
                text-align: center;
            }
            
            .empty-state-title {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 8px;
                transition: color 0.3s ease;
            }
            
            body .empty-state-title {
                color: #000000;
            }
            
            body.dark-mode .empty-state-title {
                color: #ffffff;
            }
            
            .empty-state-text {
                font-size: 14px;
                font-weight: 500;
                transition: color 0.3s ease;
            }
            
            body .empty-state-text {
                color: #737373;
            }
            
            body.dark-mode .empty-state-text {
                color: #737373;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            body ::-webkit-scrollbar-track {
                background: #fafafa;
            }
            
            body.dark-mode ::-webkit-scrollbar-track {
                background: #0f0f0f;
            }
            
            body ::-webkit-scrollbar-thumb {
                background: #d4d4d4;
                border-radius: 4px;
            }
            
            body.dark-mode ::-webkit-scrollbar-thumb {
                background: #262626;
                border-radius: 4px;
            }
            
            body ::-webkit-scrollbar-thumb:hover {
                background: #a3a3a3;
            }
            
            body.dark-mode ::-webkit-scrollbar-thumb:hover {
                background: #333333;
            }
            
            /* Dropdown */
            body .Select-control {
                background: #ffffff !important;
                border: 1px solid #e5e5e5 !important;
                border-radius: 6px !important;
                min-height: 40px !important;
            }
            
            body.dark-mode .Select-control {
                background: #0a0a0a !important;
                border: 1px solid #1a1a1a !important;
                color: #ffffff !important;
            }
            
            body .Select-control:hover {
                border-color: #d4d4d4 !important;
            }
            
            body.dark-mode .Select-control:hover {
                border-color: #262626 !important;
            }
            
            body .Select-control.is-focused {
                border-color: #000000 !important;
                box-shadow: none !important;
            }
            
            body.dark-mode .Select-control.is-focused {
                border-color: #ffffff !important;
                box-shadow: none !important;
            }
            
            body .Select-menu-outer {
                background: #ffffff !important;
                border: 1px solid #e5e5e5 !important;
                border-radius: 6px !important;
                margin-top: 4px !important;
            }
            
            body.dark-mode .Select-menu-outer {
                background: #0a0a0a !important;
                border: 1px solid #1a1a1a !important;
            }
            
            body .Select-option {
                padding: 10px 12px !important;
                background: transparent !important;
                color: #000000 !important;
            }
            
            body.dark-mode .Select-option {
                color: #ffffff !important;
            }
            
            body .Select-option.is-focused {
                background: #f5f5f5 !important;
            }
            
            body.dark-mode .Select-option.is-focused {
                background: #1a1a1a !important;
            }
            
            body .Select-option.is-selected {
                background: #fafafa !important;
                font-weight: 600 !important;
            }
            
            body.dark-mode .Select-option.is-selected {
                background: #0f0f0f !important;
                font-weight: 600 !important;
            }
            
            /* Radio Items */
            body .form-check-input {
                background-color: #ffffff !important;
                border: 1px solid #e5e5e5 !important;
            }
            
            body.dark-mode .form-check-input {
                background-color: #0a0a0a !important;
                border: 1px solid #1a1a1a !important;
            }
            
            body .form-check-input:checked {
                background-color: #000000 !important;
                border-color: #000000 !important;
            }
            
            body.dark-mode .form-check-input:checked {
                background-color: #ffffff !important;
                border-color: #ffffff !important;
            }
            
            body .form-check-label {
                color: #000000 !important;
            }
            
            body.dark-mode .form-check-label {
                color: #ffffff !important;
            }
            
            /* Switch */
            .form-switch .form-check-input {
                width: 40px !important;
                height: 20px !important;
            }
            
            /* Slider */
            body .rc-slider-track {
                background: #000000 !important;
            }
            
            body.dark-mode .rc-slider-track {
                background: #ffffff !important;
            }
            
            body .rc-slider-handle {
                border: 2px solid #000000 !important;
                background: #ffffff !important;
            }
            
            body.dark-mode .rc-slider-handle {
                border: 2px solid #ffffff !important;
                background: #0a0a0a !important;
            }
            
            body .rc-slider-rail {
                background: #e5e5e5 !important;
            }
            
            body.dark-mode .rc-slider-rail {
                background: #1a1a1a !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // Dark mode toggle
            function toggleDarkMode() {
                document.body.classList.toggle('dark-mode');
                const isDark = document.body.classList.contains('dark-mode');
                localStorage.setItem('darkMode', isDark ? 'enabled' : 'disabled');
            }
            
            // Load dark mode preference
            document.addEventListener('DOMContentLoaded', function() {
                const darkMode = localStorage.getItem('darkMode');
                if (darkMode === 'enabled') {
                    document.body.classList.add('dark-mode');
                }
                
                // Modal handling
                const observer = new MutationObserver(function() {
                    const modalOverlay = document.getElementById('modal-overlay');
                    if (modalOverlay && !modalOverlay.hasAttribute('data-listener')) {
                        modalOverlay.setAttribute('data-listener', 'true');
                        modalOverlay.addEventListener('click', function(e) {
                            if (e.target === modalOverlay || e.target.hasAttribute('data-modal-bg')) {
                                modalOverlay.classList.remove('active');
                            }
                        });
                    }
                });
                observer.observe(document.body, { childList: true, subtree: true });
            });
        </script>
    </body>
</html>
'''

# ============= HELPER FUNCTIONS =============
def get_api_stats():
    """Get statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_documents():
    """Get list of documents from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/documents",
            json={"limit": 100, "offset": 0, "bucket": "rag-documents"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except:
        return []


def upload_document(file_content, filename, config):
    """Upload document to API"""
    try:
        files = {'file': (filename, file_content, 'application/pdf')}
        data = {'config': json.dumps(config)}
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/upload",
            files=files,
            data=data,
            timeout=600
        )
        
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "Upload timeout - file too large"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def query_documents(query_config):
    """Query documents from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            json=query_config,
            timeout=180
        )
        
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "Query timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def format_sources(source_pdfs, sources):
    """Format sources dengan tampilan yang clean"""
    if not source_pdfs:
        return None
    
    source_items = []
    for pdf in source_pdfs:
        pdf_sources = [s for s in sources if s.get('metadata', {}).get('source_pdf_url') == pdf['url'] or s.get('metadata', {}).get('pdf_url') == pdf['url']]
        
        text_count = len([s for s in pdf_sources if s.get('type') == 'text'])
        image_count = len([s for s in pdf_sources if s.get('type') == 'image'])
        table_count = len([s for s in pdf_sources if s.get('type') == 'table'])
        
        counts = []
        if text_count > 0:
            counts.append(f"{text_count} text")
        if image_count > 0:
            counts.append(f"{image_count} image")
        if table_count > 0:
            counts.append(f"{table_count} table")
        
        source_item = html.Div([
            html.Div([
                html.Div(pdf.get('filename') or "Document", className="source-name"),
                html.Div(" • ".join(counts), className="source-type")
            ], className="source-header"),
            html.Div(f"Bucket: {pdf.get('bucket', 'N/A')}", className="source-meta")
        ], className="source-item", id={"type": "source-card", "index": len(source_items)}, n_clicks=0)
        
        source_items.append(source_item)
    
    return html.Div([
        html.Div(f"{len(source_pdfs)} Sources", className="sources-header"),
        html.Div(source_items)
    ], className="sources-container")


def create_modal_content(pdf_data, all_sources):
    """Create modal content"""
    pdf_url = pdf_data.get('url', '')
    
    pdf_sources = [
        s for s in all_sources 
        if s.get('metadata', {}).get('source_pdf_url') == pdf_url 
        or s.get('metadata', {}).get('pdf_url') == pdf_url
    ]
    
    text_sources = [s for s in pdf_sources if s.get('type') == 'text']
    image_sources = [s for s in pdf_sources if s.get('type') == 'image']
    table_sources = [s for s in pdf_sources if s.get('type') == 'table']
    
    content_sections = []
    
    # Text
    if text_sources:
        text_cards = []
        for idx, source in enumerate(text_sources):
            text_cards.append(
                html.Div([
                    html.Div([
                        html.Div(source.get('id', f'Text {idx+1}'), className="content-preview-title"),
                        html.Div(f"Page {source.get('page', 'N/A')} • Text", className="content-preview-meta")
                    ], className="content-preview-header"),
                    html.Div(
                        source.get('content_preview', source.get('content', 'No content'))[:300] + "...",
                        className="content-preview-text"
                    )
                ], className="content-preview-card")
            )
        
        content_sections.append(
            html.Div([
                html.Div(f"Text ({len(text_sources)})", className="modal-section-title"),
                html.Div(text_cards)
            ], className="modal-section")
        )
    
    # Images
    if image_sources:
        image_cards = []
        for idx, source in enumerate(image_sources):
            image_cards.append(
                html.Div([
                    html.Div([
                        html.Div(source.get('id', f'Image {idx+1}'), className="content-preview-title"),
                        html.Div(
                            f"Page {source.get('page', 'N/A')} • Image" + 
                            (" • Vision" if source.get('analyzed_with_vision') else ""),
                            className="content-preview-meta"
                        )
                    ], className="content-preview-header"),
                    html.Div(
                        source.get('description', 'No description')[:200] + "...",
                        className="content-preview-text",
                        style={"marginBottom": "12px"}
                    ) if source.get('description') else None,
                    html.Div([
                        html.Img(src=source.get('url', ''), style={"width": "100%", "display": "block"})
                    ], className="content-preview-image") if source.get('url') else None
                ], className="content-preview-card")
            )
        
        content_sections.append(
            html.Div([
                html.Div(f"Images ({len(image_sources)})", className="modal-section-title"),
                html.Div(image_cards)
            ], className="modal-section")
        )
    
    # Tables
    if table_sources:
        table_cards = []
        for idx, source in enumerate(table_sources):
            table_cards.append(
                html.Div([
                    html.Div([
                        html.Div(source.get('id', f'Table {idx+1}'), className="content-preview-title"),
                        html.Div(
                            f"Page {source.get('page', 'N/A')} • Table" +
                            (" • HTML" if source.get('has_html') else ""),
                            className="content-preview-meta"
                        )
                    ], className="content-preview-header"),
                    html.Div(
                        source.get('content_preview', source.get('content', 'No content'))[:300] + "...",
                        className="content-preview-text"
                    )
                ], className="content-preview-card")
            )
        
        content_sections.append(
            html.Div([
                html.Div(f"Tables ({len(table_sources)})", className="modal-section-title"),
                html.Div(table_cards)
            ], className="modal-section")
        )
    
    return html.Div([
        html.Div([
            html.Div(pdf_data.get('filename') or "Document Details", className="modal-title"),
            html.Div("×", className="modal-close", id="modal-close", n_clicks=0)
        ], className="modal-header"),
        html.Div(content_sections, className="modal-body")
    ], className="modal-content", n_clicks=0)


# ============= LAYOUT =============
app.layout = html.Div([
    dcc.Store(id='chat-history', data=[]),
    dcc.Store(id='active-page', data='chat'),
    dcc.Store(id='modal-data', data=None),
    dcc.Store(id='current-sources', data=[]),
    dcc.Store(id='current-source-pdfs', data=[]),
    dcc.Interval(id='stats-interval', interval=10000, n_intervals=0),
    
    html.Div(id="modal-overlay", className="modal-overlay", children=[
        html.Div(id="modal-content-wrapper", n_clicks=0)
    ], n_clicks=0, **{"data-modal-bg": "true"}),
    
    html.Div([
        html.Div([
            html.Div([
                html.H1("RAG System", className="sidebar-logo")
            ], className="sidebar-header"),
            
            html.Div([
                html.Div("Chat", id="menu-chat", className="menu-item active"),
                html.Div("Upload", id="menu-upload", className="menu-item"),
                html.Div("Documents", id="menu-documents", className="menu-item"),
            ], className="sidebar-menu"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Text", className="stat-mini-label"),
                        html.Div(id="stat-text-mini", className="stat-mini-value")
                    ], className="stat-mini"),
                    html.Div([
                        html.Div("Images", className="stat-mini-label"),
                        html.Div(id="stat-images-mini", className="stat-mini-value")
                    ], className="stat-mini"),
                    html.Div([
                        html.Div("Tables", className="stat-mini-label"),
                        html.Div(id="stat-tables-mini", className="stat-mini-value")
                    ], className="stat-mini"),
                ], className="stats-mini"),
                
                # Dark Mode Toggle
                html.Div([
                    html.Button("Toggle Dark Mode", className="toggle-button", id="dark-mode-toggle", n_clicks=0)
                ], className="dark-mode-toggle")
            ], className="sidebar-footer")
        ], className="sidebar"),
        
        html.Div([html.Div(id="page-content")], className="main-content")
    ], className="app-container")
])


# ============= PAGES =============
def get_chat_page():
    return html.Div([
        html.Div([
            html.Div("AI Assistant", className="header-title"),
            html.Div(id="connection-status")
        ], className="header-bar"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div(id="chat-messages", className="chat-messages", children=[
                        html.Div([
                            html.Div("Start Conversation", className="empty-state-title"),
                            html.Div("Ask anything about your documents", className="empty-state-text")
                        ], className="empty-state")
                    ]),
                    
                    html.Div([
                        html.Div([
                            dcc.Textarea(
                                id="chat-input", 
                                placeholder="Type your message...", 
                                className="chat-input", 
                                style={"height": "48px"}
                            ),
                            html.Button("→", id="send-button", className="send-button")
                        ], className="input-wrapper")
                    ], className="chat-input-container")
                ], className="chat-main")
            ], style={"flex": "1", "display": "flex"}),
            
            html.Div([
                html.Div("Settings", className="settings-title"),
                
                html.Div([
                    html.Label("Retrieval Method", className="settings-label"),
                    dcc.Dropdown(
                        id="retrieval-method", 
                        options=[
                            {"label": "All Sources", "value": "all"},
                            {"label": "Hybrid Search", "value": "hybrid"},
                            {"label": "MMR Diversity", "value": "mmr"},
                            {"label": "Text Only", "value": "text_only"},
                            {"label": "Images Only", "value": "image_only"},
                            {"label": "Tables Only", "value": "table_only"}
                        ], 
                        value="all", 
                        clearable=False
                    )
                ], className="settings-group"),
                
                html.Div([
                    html.Label("Generation Style", className="settings-label"),
                    dcc.Dropdown(
                        id="generation-method", 
                        options=[
                            {"label": "Simple Answer", "value": "simple"},
                            {"label": "With Citations", "value": "citations"},
                            {"label": "Structured Format", "value": "structured"}
                        ], 
                        value="simple", 
                        clearable=False
                    )
                ], className="settings-group"),
                
                html.Div([
                    html.Label("Number of Results", className="settings-label"),
                    dcc.Slider(
                        id="k-value", 
                        min=1, 
                        max=10, 
                        step=1, 
                        value=5, 
                        marks={i: str(i) for i in [1, 3, 5, 7, 10]}
                    )
                ], className="settings-group"),
                
                html.Div([
                    html.Label("Vision Analysis", className="settings-label"),
                    dbc.Checklist(
                        id="vision-toggle", 
                        options=[{"label": " Enable", "value": "vision"}], 
                        value=["vision"],
                        switch=True
                    )
                ], className="settings-group"),
                
                html.Div([
                    html.Label("Language", className="settings-label"),
                    dcc.RadioItems(
                        id="language-select", 
                        options=[
                            {"label": "Bahasa Indonesia", "value": "Indonesian"},
                            {"label": "English", "value": "English"}
                        ], 
                        value="Indonesian"
                    )
                ], className="settings-group")
            ], className="settings-panel")
        ], className="content-area")
    ], style={"height": "100vh", "display": "flex", "flexDirection": "column"})


def get_upload_page():
    return html.Div([
        html.Div([
            html.Div("Upload Document", className="header-title")
        ], className="header-bar"),
        
        html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-document', 
                    children=html.Div([
                        html.H3("Drop PDF Here", className="upload-title"),
                        html.P("or click to browse", className="upload-subtitle")
                    ]), 
                    className="upload-area",
                    multiple=False
                ),
                html.Div(id='upload-status')
            ], className="upload-container")
        ], className="content-area")
    ], style={"height": "100vh", "display": "flex", "flexDirection": "column"})


def get_documents_page():
    return html.Div([
        html.Div([
            html.Div("Document Library", className="header-title")
        ], className="header-bar"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Your Documents", className="documents-title"),
                    html.P("All documents in vector database", className="documents-subtitle")
                ], className="documents-header"),
                html.Div(id="documents-list")
            ], className="documents-container")
        ], className="content-area")
    ], style={"height": "100vh", "display": "flex", "flexDirection": "column"})


# ============= CALLBACKS =============

# Dark mode toggle (client-side callback via JavaScript in HTML)
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            toggleDarkMode();
        }
        return '';
    }
    """,
    Output("dark-mode-toggle", "children"),
    Input("dark-mode-toggle", "n_clicks"),
    prevent_initial_call=True
)


@app.callback(
    [Output("page-content", "children"),
     Output("menu-chat", "className"),
     Output("menu-upload", "className"),
     Output("menu-documents", "className"),
     Output("active-page", "data")],
    [Input("menu-chat", "n_clicks"),
     Input("menu-upload", "n_clicks"),
     Input("menu-documents", "n_clicks")],
    prevent_initial_call=True
)
def navigate(c1, c2, c3):
    tid = ctx.triggered_id
    classes = {
        "menu-chat": "menu-item",
        "menu-upload": "menu-item",
        "menu-documents": "menu-item"
    }
    
    if tid == "menu-upload":
        classes["menu-upload"] += " active"
        return (
            get_upload_page(),
            classes["menu-chat"],
            classes["menu-upload"],
            classes["menu-documents"],
            "upload"
        )
    elif tid == "menu-documents":
        classes["menu-documents"] += " active"
        return (
            get_documents_page(),
            classes["menu-chat"],
            classes["menu-upload"],
            classes["menu-documents"],
            "documents"
        )
    else:
        classes["menu-chat"] += " active"
        return (
            get_chat_page(),
            classes["menu-chat"],
            classes["menu-upload"],
            classes["menu-documents"],
            "chat"
        )


@app.callback(
    [Output("stat-text-mini", "children"),
     Output("stat-images-mini", "children"),
     Output("stat-tables-mini", "children"),
     Output("connection-status", "children")],
    Input("stats-interval", "n_intervals")
)
def update_stats(n):
    stats = get_api_stats()
    
    if stats and stats.get("status") == "success":
        collections = stats.get("collections", {})
        status = html.Div([
            html.Div(className="status-dot"),
            "Connected"
        ], className="status-indicator")
        
        return (
            str(collections.get("text", 0)),
            str(collections.get("images", 0)),
            str(collections.get("tables", 0)),
            status
        )
    
    status = html.Div([
        html.Div(className="status-dot"),
        "Disconnected"
    ], className="status-indicator disconnected")
    
    return "0", "0", "0", status


@app.callback(
    Output("documents-list", "children"),
    Input("stats-interval", "n_intervals")
)
def update_documents_list(n):
    docs = get_documents()
    
    if not docs:
        return html.Div([
            html.Div("No Documents", className="empty-state-title"),
            html.Div("Upload a PDF to get started", className="empty-state-text")
        ], className="empty-state")
    
    return html.Div([
        html.Div([
            html.Div(doc.get("name", "Unknown"), className="document-name"),
            html.Div([
                html.Span(f"{doc.get('size_mb', 0):.2f} MB"),
                html.Span("•"),
                html.Span(doc.get('created_at', 'N/A')[:10])
            ], className="document-meta")
        ], className="document-card") for doc in docs
    ], className="documents-grid")


@app.callback(
    [Output("upload-status", "children"),
     Output("stats-interval", "n_intervals", allow_duplicate=True)],
    Input("upload-document", "contents"),
    State("upload-document", "filename"),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents:
        raise PreventUpdate
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        config = {
            "extract_images": True,
            "extract_tables": True,
            "generate_image_descriptions": True,
            "description_language": "Indonesian",
            "vision_model": "gpt-4o-mini",
            "store_to_vectordb": True,
            "strategy": "hi_res",
            "upload_source_pdf": True,
            "custom_pdf_filename": None
        }
        
        result = upload_document(decoded, filename, config)
        
        if result.get("status") == "success":
            stats = result.get("stats", {})
            
            badges = [
                html.Span(f"{stats.get('text_chunks', 0)} text", className="badge badge-success", style={"margin": "4px"}),
                html.Span(f"{stats.get('images', 0)} images", className="badge badge-success", style={"margin": "4px"}),
                html.Span(f"{stats.get('tables', 0)} tables", className="badge badge-success", style={"margin": "4px"}),
            ]
            
            return html.Div([
                html.H3("Upload Successful", style={"marginBottom": "16px", "fontSize": "20px", "fontWeight": "600"}),
                html.Div(badges)
            ], style={"textAlign": "center", "padding": "48px"}), 1
        else:
            return html.Div([
                html.H3("Upload Failed", style={"marginBottom": "8px", "fontSize": "20px", "fontWeight": "600"}),
                html.P(result.get('error', 'Unknown error'), style={"fontSize": "14px"})
            ], style={"textAlign": "center", "padding": "48px"}), 0
            
    except Exception as e:
        return html.Div([
            html.H3("Error Occurred", style={"marginBottom": "8px", "fontSize": "20px", "fontWeight": "600"}),
            html.P(str(e), style={"fontSize": "14px"})
        ], style={"textAlign": "center", "padding": "48px"}), 0


@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-input", "value"),
     Output("chat-history", "data"),
     Output("current-sources", "data"),
     Output("current-source-pdfs", "data")],
    Input("send-button", "n_clicks"),
    [State("chat-input", "value"),
     State("retrieval-method", "value"),
     State("generation-method", "value"),
     State("k-value", "value"),
     State("vision-toggle", "value"),
     State("language-select", "value"),
     State("chat-history", "data")],
    prevent_initial_call=True
)
def handle_chat(n, query, retrieval_method, generation_method, k_value, vision_toggle, language, chat_history):
    if not query or query.strip() == "":
        raise PreventUpdate
    
    timestamp = datetime.now().strftime("%H:%M")
    user_message = {
        "role": "user",
        "content": query,
        "timestamp": timestamp
    }
    
    chat_history = chat_history or []
    chat_history.append(user_message)
    
    use_vision = "vision" in (vision_toggle or [])
    
    query_config = {
        "query": query,
        "retrieval_method": retrieval_method,
        "generation_method": generation_method,
        "language": language,
        "k": k_value,
        "k_text": k_value,
        "k_images": max(2, k_value // 2),
        "k_tables": max(2, k_value // 2),
        "include_sources": True,
        "include_source_pdfs": True,
        "use_vision": use_vision
    }
    
    result = query_documents(query_config)
    
    if result.get("status") == "success":
        assistant_message = {
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
            "source_pdfs": result.get("source_pdfs", []),
            "timestamp": datetime.now().strftime("%H:%M"),
            "vision_used": result.get("vision_used", False),
            "processing_time": result.get("processing_time_seconds", 0)
        }
    else:
        assistant_message = {
            "role": "assistant",
            "content": f"Error: {result.get('error', 'Unknown error')}",
            "sources": [],
            "source_pdfs": [],
            "timestamp": datetime.now().strftime("%H:%M")
        }
    
    chat_history.append(assistant_message)
    
    messages_div = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages_div.append(
                html.Div([
                    html.Div([
                        html.Div(msg["content"], className="message-bubble"),
                        html.Div(msg["timestamp"], className="message-meta")
                    ], className="message-wrapper")
                ], className="message message-user")
            )
        else:
            # Images section
            image_sources = [s for s in msg.get("sources", []) if s.get("type") == "image" and s.get("url")]
            images_section = None
            
            if image_sources:
                image_cards = []
                for idx, img_source in enumerate(image_sources):
                    image_cards.append(
                        html.Div([
                            html.Div([
                                html.Img(src=img_source.get("url", ""), style={"width": "100%", "height": "100%", "objectFit": "cover"})
                            ], className="image-wrapper"),
                            html.Div(f"Page {img_source.get('page', 'N/A')}", className="image-label")
                        ], className="image-card")
                    )
                
                images_section = html.Div([
                    html.Div(f"{len(image_sources)} Images", className="images-header"),
                    html.Div(image_cards, className="images-grid")
                ], className="images-section")
            
            # Sources
            sources_div = format_sources(msg.get("source_pdfs", []), msg.get("sources", []))
            
            badges = []
            if msg.get("vision_used"):
                badges.append(html.Span("Vision", className="badge badge-success", style={"marginLeft": "8px"}))
            
            messages_div.append(
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div(msg["content"], style={"whiteSpace": "pre-wrap"}),
                            images_section if images_section else None,
                            sources_div if sources_div else None
                        ], className="message-bubble"),
                        html.Div([
                            html.Span(msg["timestamp"]),
                            *badges
                        ], className="message-meta")
                    ], className="message-wrapper")
                ], className="message message-assistant")
            )
    
    last_msg = chat_history[-1] if chat_history else {}
    current_sources = last_msg.get("sources", [])
    current_source_pdfs = last_msg.get("source_pdfs", [])
    
    return messages_div, "", chat_history, current_sources, current_source_pdfs


@app.callback(
    [Output("modal-overlay", "className"),
     Output("modal-content-wrapper", "children")],
    Input({"type": "source-card", "index": ALL}, "n_clicks"),
    [State("current-source-pdfs", "data"),
     State("current-sources", "data")],
    prevent_initial_call=True
)
def handle_source_click(n_clicks, source_pdfs, sources):
    if not any(n_clicks) or not source_pdfs:
        raise PreventUpdate
    
    triggered = ctx.triggered[0]
    if not triggered["value"]:
        raise PreventUpdate
    
    import re
    match = re.search(r'"index":(\d+)', triggered["prop_id"])
    if not match:
        raise PreventUpdate
    
    source_index = int(match.group(1))
    
    if source_index >= len(source_pdfs):
        raise PreventUpdate
    
    pdf_data = source_pdfs[source_index]
    modal_content = create_modal_content(pdf_data, sources)
    
    return "modal-overlay active", modal_content


@app.callback(
    Output("modal-overlay", "className", allow_duplicate=True),
    Input("modal-close", "n_clicks"),
    prevent_initial_call=True
)
def close_modal(close_clicks):
    if close_clicks:
        return "modal-overlay"
    raise PreventUpdate


# ============= RUN APP =============
if __name__ == '__main__':
    print("\n" + "="*60)
    print("MULTIMODAL RAG SYSTEM - WITH DARK MODE")
    print("="*60)
    print("Dashboard: http://localhost:8050")
    print("Dark Mode: Click 'Toggle Dark Mode' button")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8050)