# **Edge & Blur Detection Model**  

## **Goal**  
This model applies **edge detection** and **blurring** to input images using **convolutional filters**. It identifies vertical, horizontal, and diagonal edges and can smooth images using a blur filter. This is useful for **image processing** and **computer vision applications**.  

---

## **Concepts Used**  

### **1️⃣ Convolutional Filters (Kernels)**  
A **kernel** (or filter) is a small matrix applied over an image to detect features such as edges or blur. The model uses the following 3×3 kernels:  

- **Vertical Edge Detection:**  

### **2️⃣ Stride**  
The **stride** defines how much the kernel moves across the image. A **stride of 1** means the filter moves one pixel at a time, preserving more details. Higher strides reduce the size of the output image.  

- **In this model:** We use `stride = 1` to ensure fine-grained edge detection.

### **3️⃣ Padding**  
Padding determines how borders are handled:  
- **"SAME" Padding** – Keeps the image size unchanged by adding zeros around the edges.  
- **"VALID" Padding** – Reduces image size by applying the filter without extra padding.  

- **In this model:** We use **"SAME" padding** so the output image size remains the same.

---

## **Technologies Used**  
- **TensorFlow** – For defining and applying convolutional filters.  
- **NumPy** – For numerical operations and array manipulation.  
- **Pillow (PIL)** – For image loading and preprocessing.  
- **Matplotlib** – For visualizing processed images.  

---
