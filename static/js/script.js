// DocuSum - Interactive JavaScript for Document Summarizer ChatBot

// Global variables
let uploadedFile = null;
let currentFileId = null;
let isProcessing = false;
let chatHistory = [];
let documentText = null;

// DOM Elements
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const uploadProgress = document.getElementById("uploadProgress");
const documentInfo = document.getElementById("documentInfo");
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const typingIndicator = document.getElementById("typingIndicator");
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
  setupEventListeners();
  updateNavOnScroll();
});

// Initialize application
function initializeApp() {
  // Set initial status
  updateStatus("ready", "Ready to help");

  // Add welcome message animation
  setTimeout(() => {
    const welcomeMessage = document.querySelector(".assistant-message");
    if (welcomeMessage) {
      welcomeMessage.style.animation = "messageSlide 0.6s ease-out";
    }
  }, 500);

  // Initialize smooth scrolling for navigation
  initializeSmoothScrolling();

  // Add intersection observer for animations
  initializeAnimations();
}

// Setup event listeners
function setupEventListeners() {
  // File upload events
  uploadArea.addEventListener("click", () => fileInput.click());
  uploadArea.addEventListener("dragover", handleDragOver);
  uploadArea.addEventListener("dragleave", handleDragLeave);
  uploadArea.addEventListener("drop", handleDrop);
  fileInput.addEventListener("change", handleFileSelect);

  // Chat events
  chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener("click", sendMessage);

  // Navigation toggle for mobile
  const navToggle = document.querySelector(".nav-toggle");
  const navMenu = document.querySelector(".nav-menu");

  if (navToggle && navMenu) {
    navToggle.addEventListener("click", () => {
      navMenu.classList.toggle("active");
    });
  }

  // Close mobile menu on link click
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      if (navMenu) {
        navMenu.classList.remove("active");
      }
    });
  });
}

// File upload handlers
function handleDragOver(e) {
  e.preventDefault();
  uploadArea.classList.add("dragover");
}

function handleDragLeave(e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
}

function handleDrop(e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    handleFile(file);
  }
}

function handleFile(file) {
  // Validate file type
  if (file.type !== "application/pdf") {
    showNotification("Please select a PDF file.", "error");
    return;
  }

  // Validate file size (10MB limit)
  if (file.size > 10 * 1024 * 1024) {
    showNotification("File size must be less than 10MB.", "error");
    return;
  }

  uploadedFile = file;
  showUploadProgress();
  uploadFileToServer(file);
}

function showUploadProgress() {
  uploadArea.style.display = "none";
  uploadProgress.style.display = "block";
  documentInfo.style.display = "none";
}

async function uploadFileToServer(file) {
  const progressFill = document.querySelector(".progress-fill");
  const progressText = document.querySelector(".progress-text");

  updateStatus("processing", "Uploading and processing document...");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (result.success) {
      currentFileId = result.file_id;
      documentText = result.preview;

      // Simulate progress for visual feedback
      let progress = 0;
      const interval = setInterval(() => {
        progress += 20;
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          setTimeout(() => {
            showDocumentInfo(file);
            updateStatus("ready", "Document ready for analysis");
          }, 300);
        }
        progressFill.style.width = progress + "%";
        progressText.textContent = `Processing... ${Math.round(progress)}%`;
      }, 100);
    } else {
      throw new Error(result.error || "Upload failed");
    }
  } catch (error) {
    console.error("Upload error:", error);
    showNotification("Upload failed: " + error.message, "error");
    resetUploadArea();
    updateStatus("ready", "Ready to help");
  }
}

function showDocumentInfo(file) {
  uploadProgress.style.display = "none";
  documentInfo.style.display = "block";

  // Update document details
  document.getElementById("docName").textContent = file.name;
  document.getElementById("docSize").textContent =
    `Size: ${formatFileSize(file.size)}`;

  // Enable chat input
  chatInput.disabled = false;
  sendBtn.disabled = false;
  chatInput.placeholder = "Ask a question about your document...";

  // Add document uploaded message to chat
  addMessage(
    "assistant",
    `Great! I've processed "${file.name}". You can now ask me questions about the document or request a summary.`,
  );
}

function removeDocument() {
  uploadedFile = null;
  currentFileId = null;
  documentText = null;

  resetUploadArea();
  clearChat();
  updateStatus("ready", "Ready to help");
  showNotification("Document removed successfully.", "success");
}

function resetUploadArea() {
  // Reset UI
  uploadArea.style.display = "block";
  uploadProgress.style.display = "none";
  documentInfo.style.display = "none";

  // Disable chat
  chatInput.disabled = true;
  sendBtn.disabled = true;
  chatInput.placeholder = "Upload a document to start chatting...";

  // Reset file input
  fileInput.value = "";
}

// Chat functionality
async function sendMessage() {
  const message = chatInput.value.trim();
  if (!message || isProcessing) return;

  // Add user message
  addMessage("user", message);
  chatInput.value = "";

  // Show typing indicator
  showTypingIndicator();
  updateStatus("processing", "Thinking...");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        file_id: currentFileId,
      }),
    });

    const result = await response.json();

    if (result.success) {
      hideTypingIndicator();
      addMessage("assistant", result.response);
      updateStatus("ready", "Ready to help");
    } else {
      throw new Error(result.error || "Chat request failed");
    }
  } catch (error) {
    console.error("Chat error:", error);
    hideTypingIndicator();
    addMessage(
      "assistant",
      "I apologize, but I encountered an error. Please try again.",
    );
    updateStatus("ready", "Ready to help");
    showNotification("Error: " + error.message, "error");
  }
}

function addMessage(type, content) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${type}-message`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.innerHTML =
    type === "assistant"
      ? '<i class="fas fa-robot"></i>'
      : '<i class="fas fa-user"></i>';

  const messageContent = document.createElement("div");
  messageContent.className = "message-content";
  messageContent.innerHTML = `<p>${content}</p>`;

  const messageTime = document.createElement("div");
  messageTime.className = "message-time";
  messageTime.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(messageContent);
  messageContent.appendChild(messageTime);

  chatMessages.appendChild(messageDiv);
  scrollChatToBottom();

  // Store in history
  chatHistory.push({ type, content, timestamp: new Date() });
}

function showTypingIndicator() {
  typingIndicator.style.display = "flex";
  scrollChatToBottom();
  isProcessing = true;
}

function hideTypingIndicator() {
  typingIndicator.style.display = "none";
  isProcessing = false;
}

function scrollChatToBottom() {
  setTimeout(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 100);
}

async function askQuickQuestion(question) {
  if (isProcessing || !currentFileId) return;

  chatInput.value = question;
  await sendMessage();
}

// This function is now defined above, removing duplicate

function clearChat() {
  // Keep only the welcome message
  const welcomeMessage = chatMessages.querySelector(".assistant-message");
  chatMessages.innerHTML = "";
  if (welcomeMessage) {
    chatMessages.appendChild(welcomeMessage);
  }
  chatHistory = [];
}

// Document processing functions
async function summarizeDocument() {
  if (!currentFileId || isProcessing) return;

  const summarizeBtn = document.getElementById("summarizeBtn");
  const originalText = summarizeBtn.innerHTML;

  summarizeBtn.innerHTML =
    '<i class="fas fa-spinner fa-spin"></i> Summarizing...';
  summarizeBtn.disabled = true;

  updateStatus("processing", "Generating summary...");

  try {
    const response = await fetch("/summarize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        file_id: currentFileId,
      }),
    });

    const result = await response.json();

    if (result.success) {
      const summaryMessage = `Here's a comprehensive summary of your document:\n\n${result.summary}\n\nOriginal length: ${result.original_length} characters\nSummary length: ${result.summary_length} characters\n\nWould you like me to elaborate on any specific section?`;

      addMessage("assistant", summaryMessage);
      updateStatus("ready", "Summary generated");
    } else {
      throw new Error(result.error || "Summarization failed");
    }
  } catch (error) {
    console.error("Summarization error:", error);
    addMessage(
      "assistant",
      "I apologize, but I encountered an error while generating the summary. Please try again.",
    );
    showNotification("Error: " + error.message, "error");
    updateStatus("ready", "Ready to help");
  } finally {
    summarizeBtn.innerHTML = originalText;
    summarizeBtn.disabled = false;

    setTimeout(() => {
      updateStatus("ready", "Ready to help");
    }, 2000);
  }
}

// Utility functions
function updateStatus(type, message) {
  statusText.textContent = message;

  switch (type) {
    case "ready":
      statusIndicator.style.background = "#10b981";
      break;
    case "processing":
      statusIndicator.style.background = "#f59e0b";
      break;
    case "error":
      statusIndicator.style.background = "#ef4444";
      break;
  }
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function showNotification(message, type = "info") {
  // Create notification element
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === "success" ? "check-circle" : type === "error" ? "exclamation-circle" : "info-circle"}"></i>
            <span>${message}</span>
        </div>
    `;

  // Add styles for notification
  notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        padding: 1rem;
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;

  document.body.appendChild(notification);

  // Animate in
  setTimeout(() => {
    notification.style.transform = "translateX(0)";
  }, 100);

  // Remove after 3 seconds
  setTimeout(() => {
    notification.style.transform = "translateX(100%)";
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 300);
  }, 3000);
}

function scrollToUpload() {
  document.getElementById("app-interface").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

// Navigation and scrolling
function updateNavOnScroll() {
  const navbar = document.querySelector(".navbar");
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-link");

  window.addEventListener("scroll", () => {
    // Update navbar background
    if (window.scrollY > 100) {
      navbar.style.background = "rgba(255, 255, 255, 0.98)";
      navbar.style.boxShadow = "0 4px 20px rgba(0, 0, 0, 0.1)";
    } else {
      navbar.style.background = "rgba(255, 255, 255, 0.95)";
      navbar.style.boxShadow = "none";
    }

    // Update active navigation
    let current = "";
    sections.forEach((section) => {
      const sectionTop = section.offsetTop - 150;
      const sectionHeight = section.offsetHeight;
      if (
        window.scrollY >= sectionTop &&
        window.scrollY < sectionTop + sectionHeight
      ) {
        current = section.getAttribute("id");
      }
    });

    navLinks.forEach((link) => {
      link.classList.remove("active");
      if (link.getAttribute("href") === "#" + current) {
        link.classList.add("active");
      }
    });
  });
}

function initializeSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
}

function initializeAnimations() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1";
        entry.target.style.transform = "translateY(0)";
      }
    });
  }, observerOptions);

  // Observe elements for animation
  document.querySelectorAll(".feature-card, .step, .card").forEach((el) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(30px)";
    el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
    observer.observe(el);
  });
}

// Add CSS for notifications if not already present
const notificationStyles = `
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .notification-success {
        border-left: 4px solid #10b981;
        color: #059669;
    }

    .notification-error {
        border-left: 4px solid #ef4444;
        color: #dc2626;
    }

    .notification-info {
        border-left: 4px solid #3b82f6;
        color: #2563eb;
    }
`;

// Inject notification styles
const styleSheet = document.createElement("style");
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Add mobile navigation styles
const mobileNavStyles = `
    @media (max-width: 768px) {
        .nav-menu {
            position: fixed;
            top: 80px;
            right: -100%;
            width: 300px;
            height: calc(100vh - 80px);
            background: white;
            flex-direction: column;
            justify-content: flex-start;
            align-items: center;
            padding-top: 2rem;
            box-shadow: -5px 0 20px rgba(0,0,0,0.1);
            transition: right 0.3s ease;
            z-index: 999;
        }

        .nav-menu.active {
            right: 0;
        }

        .nav-toggle.active span:nth-child(1) {
            transform: rotate(45deg) translate(5px, 5px);
        }

        .nav-toggle.active span:nth-child(2) {
            opacity: 0;
        }

        .nav-toggle.active span:nth-child(3) {
            transform: rotate(-45deg) translate(7px, -6px);
        }
    }
`;

const mobileStyleSheet = document.createElement("style");
mobileStyleSheet.textContent = mobileNavStyles;
document.head.appendChild(mobileStyleSheet);

// Export functions for global access
window.DocuSum = {
  scrollToUpload,
  summarizeDocument,
  removeDocument,
  sendMessage,
  askQuickQuestion,
};

console.log("DocuSum Application Loaded Successfully! 🤖");
