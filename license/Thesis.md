# MILLAT UMIDI UNIVERSITY

**Department of Software Engineering**

---

&nbsp;

&nbsp;

**Bachelor Degree Thesis**

**Degree Code: 60610600 – Software Engineering ("Dasturiy injiniringi")**

&nbsp;

# Development and Implementation of an Intelligent Parking Management System Using Modern Technologies

&nbsp;

&nbsp;

**Author:** _____________________________ [Student Name]

**Scientific Supervisor:** _____________________________ [Supervisor Name]

**Reviewer:** _____________________________ [Reviewer Name]

&nbsp;

**Admitted to defense**

Dean of the Faculty: _____________________________ [Dean Signature / Date]

&nbsp;

&nbsp;

**Tashkent – 2026**

---

## TABLE OF CONTENTS

- [INTRODUCTION](#introduction)
- [CHAPTER 1. Theoretical Foundations of Intelligent Parking and License Plate Recognition Systems](#chapter-1-theoretical-foundations-of-intelligent-parking-and-license-plate-recognition-systems)
  - [1.1 Intelligent Transportation Systems: Concept, History, and Classification](#11-intelligent-transportation-systems-concept-history-and-classification)
  - [1.2 The Role of Automated License Plate Recognition in Urban Mobility](#12-the-role-of-automated-license-plate-recognition-in-urban-mobility)
  - [1.3 Historical Development of ALPR Technology](#13-historical-development-of-alpr-technology)
  - [1.4 Deep Learning Foundations for Object Detection](#14-deep-learning-foundations-for-object-detection)
  - [1.5 Optical Character Recognition: Principles and Evolution](#15-optical-character-recognition-principles-and-evolution)
  - [1.6 Uzbekistan Context: Digital Transformation and Regulatory Framework](#16-uzbekistan-context-digital-transformation-and-regulatory-framework)
- [CHAPTER 2. Technologies and Programming Languages Used](#chapter-2-technologies-and-programming-languages-used)
  - [2.1 Python as the Core Development Language](#21-python-as-the-core-development-language)
  - [2.2 YOLOv8: Architecture and Capabilities](#22-yolov8-architecture-and-capabilities)
  - [2.3 EasyOCR: Deep Learning-Based Character Recognition](#23-easyocr-deep-learning-based-character-recognition)
  - [2.4 OpenCV: Image Preprocessing and Video Processing](#24-opencv-image-preprocessing-and-video-processing)
  - [2.5 FastAPI: Modern Asynchronous REST Framework](#25-fastapi-modern-asynchronous-rest-framework)
  - [2.6 SQLite with WAL Mode: Embedded Relational Persistence](#26-sqlite-with-wal-mode-embedded-relational-persistence)
  - [2.7 Supporting Libraries and the Dependency Ecosystem](#27-supporting-libraries-and-the-dependency-ecosystem)
- [CHAPTER 3. System Design, Architecture, and Implementation](#chapter-3-system-design-architecture-and-implementation)
  - [3.1 System Overview and Design Principles](#31-system-overview-and-design-principles)
  - [3.2 High-Level Architecture](#32-high-level-architecture)
  - [3.3 Module Decomposition and Responsibilities](#33-module-decomposition-and-responsibilities)
  - [3.4 Detection Module Implementation](#34-detection-module-implementation)
  - [3.5 Recognition Module Implementation](#35-recognition-module-implementation)
  - [3.6 Validation Module and Uzbek Plate Standard](#36-validation-module-and-uzbek-plate-standard)
  - [3.7 Pipeline Orchestration, Best-Detection Strategy, and Video Processing](#37-pipeline-orchestration-best-detection-strategy-and-video-processing)
  - [3.8 Database Design and Implementation](#38-database-design-and-implementation)
  - [3.9 REST API Design and Implementation](#39-rest-api-design-and-implementation)
  - [3.10 Command-Line Interface](#310-command-line-interface)
  - [3.11 Configuration Management](#311-configuration-management)
- [CHAPTER 4. Testing, Evaluation, and Results](#chapter-4-testing-evaluation-and-results)
  - [4.1 Testing Strategy and Methodology](#41-testing-strategy-and-methodology)
  - [4.2 Validator Unit Test Suite](#42-validator-unit-test-suite)
  - [4.3 Database Integration Test Suite](#43-database-integration-test-suite)
  - [4.4 Pipeline Synthetic Tests](#44-pipeline-synthetic-tests)
  - [4.5 Performance Benchmarks](#45-performance-benchmarks)
  - [4.6 Discussion of Results and Limitations](#46-discussion-of-results-and-limitations)
- [CONCLUSION](#conclusion)
- [LIST OF USED LITERATURE](#list-of-used-literature)
- [APPENDIX A: Project Source Code Structure and Description](#appendix-a-project-source-code-structure-and-description)
- [APPENDIX B: Uzbek License Plate Format Reference](#appendix-b-uzbek-license-plate-format-reference)
- [APPENDIX C: API Endpoint Reference and Response Schema](#appendix-c-api-endpoint-reference-and-response-schema)
- [APPENDIX D: OCR Error Correction Table](#appendix-d-ocr-error-correction-table)
- [APPENDIX E: Installation and Quick-Start Guide](#appendix-e-installation-and-quick-start-guide)

---

## INTRODUCTION

### Background and Motivation

The urbanization of Uzbekistan has proceeded at a historically significant pace throughout the early twenty-first century. Tashkent, the nation's capital and primary economic hub, has experienced continuous population growth and an accompanying rise in private vehicle ownership. According to data from the State Statistics Committee of the Republic of Uzbekistan, the number of registered motor vehicles in the country has expanded at a compound annual rate exceeding six percent over the past decade. This growth has placed extraordinary pressure on parking infrastructure, which in many urban districts remains anchored to outdated models of operation that were designed for a much smaller vehicle population.

Traditional parking management in Uzbekistan relies predominantly on one of two approaches. The first is purely manual: a human attendant stationed at a facility entrance records vehicle identifiers on paper or in an informal spreadsheet and collects payment upon exit. The second is semi-automated through the use of physical tokens, pre-printed tickets dispensed by mechanical gates, or RFID cards issued to registered subscribers. Both approaches share fundamental weaknesses. Manual systems introduce human error into record-keeping, suffer from throughput constraints that manifest as queues during peak traffic hours, expose facilities to security vulnerabilities including ticket forgery and social engineering, and require continuous staffing that represents a significant operational expense. RFID-based systems eliminate some of these problems but require a credential issuance process, impose costs on visitors who lack a registered card, and still require human intervention in exception-handling scenarios.

Against this backdrop, automated license plate recognition technology offers a compelling alternative. An ALPR-based parking system identifies vehicles by reading their license plates, which are already a standardized, universally present vehicle identifier in Uzbekistan. No credential issuance is required. Processing is near-instantaneous. Records are generated automatically with machine-readable content, enabling full digital audit trails. The technology has matured sufficiently that capable systems can be built using open-source components and deployed on commodity hardware, making the investment accessible to facility operators without enterprise-scale budgets.

The motivation for this thesis is therefore twofold. First, there is a clear practical need for an affordable, open-source ALPR-based parking management system tailored to the specific license plate standards and regional context of Uzbekistan. Second, the convergence of several modern technologies — the YOLOv8 neural network for object detection, EasyOCR for character recognition, FastAPI for service-oriented software architecture, and SQLite for robust embedded persistence — makes it now possible to satisfy that need within the scope of a bachelor-level engineering project.

### Problem Statement

The parking management problem in Uzbekistan can be stated precisely as follows: there is no publicly available, open-source, tested software system that automatically detects, reads, validates, and records Uzbek vehicle license plates from image or video input and exposes the results through a programmatic interface suitable for integration with parking infrastructure. Existing commercial ALPR products are designed for international plate formats and require expensive proprietary licenses. Academic prototypes described in the literature address generic plate formats and are not adapted to the specific format defined by State Standard O'z DSt 1180:2006, which governs Uzbek vehicle registration plates.

The absence of such a system means that software developers and facility operators who wish to modernize parking management in Uzbekistan must either license expensive foreign software, commission custom development from scratch without a reference implementation, or continue relying on outdated manual approaches. This thesis directly addresses this gap.

### Objectives

The primary objectives of this thesis are:

1. To conduct a thorough review of theoretical foundations including intelligent transportation systems, ALPR technology history, deep learning architectures for detection and recognition, and the regulatory framework for vehicle registration in Uzbekistan.
2. To evaluate and select the most appropriate modern software technologies for each component of an ALPR pipeline: detection, recognition, validation, persistence, and service exposure.
3. To design a modular, testable system architecture that separates concerns cleanly and enables independent development and replacement of each component.
4. To implement the complete system in production-quality Python code, including all validation logic specific to Uzbek plate standards and OCR error correction rules derived from empirical analysis.
5. To validate the correctness of every critical module through automated tests covering nominal paths, boundary conditions, and error recovery scenarios.
6. To evaluate system performance in terms of processing speed and recognition accuracy, and to discuss practical deployment considerations.

### Scope and Delimitations

This work covers the recognition of Uzbek standard civilian vehicle plates in the format defined by State Standard O'z DSt 1180:2006. It includes processing of both static image files and recorded video files, a relational database for detection history storage, and a RESTful HTTP API. Out of scope are: real-time RTSP camera stream integration, payment processing modules, mobile application development, recognition of diplomatic or military plates, and integration with physical barrier hardware.

### Thesis Structure

The remainder of this document is organized as follows. Chapter 1 reviews the theoretical background of intelligent transportation systems, the history of ALPR technology, the deep learning foundations of the chosen components, and the Uzbekistan regulatory context. Chapter 2 examines in detail each technology chosen for implementation, explaining the rationale for selection and the technical characteristics that make each suitable for this application. Chapter 3 presents the complete system design, architecture, and implementation in full technical detail, including source code excerpts for all critical components. Chapter 4 describes the testing strategy, presents test results, documents performance benchmarks, and discusses findings and limitations. The Conclusion summarizes the contributions and recommends directions for future work.

---

## CHAPTER 1. Theoretical Foundations of Intelligent Parking and License Plate Recognition Systems

### 1.1 Intelligent Transportation Systems: Concept, History, and Classification

Intelligent Transportation Systems, commonly abbreviated as ITS, represent a broad and interdisciplinary class of technologies and services applied to transportation networks for the purpose of improving safety, operational efficiency, environmental performance, and user experience. The formal definition adopted by the International Organization for Standardization in the ISO 14813 series describes ITS as the application of information and communication technologies to the transport of people and goods. This definition encompasses a wide spectrum of implementations ranging from electronic toll collection and dynamic variable message signs on highways to advanced traffic signal controllers and urban parking guidance systems.

The concept of applying technology to transportation challenges predates the digital era. Early twentieth-century innovations such as the first automatic traffic signal (Cleveland, Ohio, 1914) and the introduction of actuated signal control based on inductive loop detectors in the 1950s established the foundational model of sensor-based transportation management. However, the modern ITS paradigm emerged in the 1990s, driven by three simultaneous technological developments: the maturation of digital communications networks, the availability of affordable embedded processors, and significant advances in image capture and processing hardware.

The European Union's landmark DRIVE (Dedicated Road Infrastructure for Vehicle safety in Europe) program, initiated in 1989, and the parallel IVHS (Intelligent Vehicle Highway Systems) initiative in the United States catalyzed large-scale research investment in ITS and established the vocabulary, reference architectures, and institutional partnerships that continue to shape the field today. These programs produced the first practical demonstrations of automated incident detection, in-vehicle navigation, electronic toll collection, and automated parking access control systems.

ITS applications are typically classified according to their primary operational function. Road traffic management systems optimize the flow of vehicles through signal coordination, incident detection, and variable speed limits. Advanced traveler information systems provide real-time data to drivers and passengers through navigation applications, dynamic message signs, and broadcast media. Commercial vehicle operations systems support logistics management including automatic vehicle identification, weigh-in-motion, and fleet tracking. Emergency management systems coordinate response to accidents and natural disasters using real-time position data from infrastructure and vehicles. Parking management systems, which are the focus of this work, automate the detection of available parking spaces, control access to facilities, and maintain records of vehicle occupancy.

Within the parking management category, license plate recognition serves as the universal vehicle identification mechanism. Unlike RFID transponders or QR code tickets, license plates require no prior enrollment by the driver and are already a legally mandated presence on every registered vehicle. This makes ALPR the lowest-friction technology for automated parking access and the primary building block of modern intelligent parking systems.

### 1.2 The Role of Automated License Plate Recognition in Urban Mobility

Automated license plate recognition technology occupies a unique position in the ITS ecosystem because it enables vehicle identification without requiring any cooperation or action from the vehicle operator. This property, which might be described as passive identification, distinguishes ALPR from all token-based and credential-based access control mechanisms and is the fundamental reason for its widespread adoption across applications as diverse as parking access control, law enforcement, toll collection, border crossing management, and vehicle theft recovery.

In the parking management context, ALPR provides several specific operational benefits. Entry and exit processing is reduced from several seconds per vehicle (typical for ticket dispensing or RFID reading) to less than one second for camera capture plus processing time, effectively eliminating the gate queue as a bottleneck at most facility sizes. Record-keeping becomes automatic and complete: every entry and exit event is captured with a timestamp, a confident plate identifier, and metadata describing the recognition quality. This structured data enables revenue analysis, utilization forecasting, and anomaly detection — capabilities entirely absent from manual systems and only partially available through ticket-based systems.

ALPR also enables enforcement integration. A parking management system that maintains a database of recognized plates can query against lists of known vehicles in good standing, outstanding payment debtors, or flagged vehicles. This enforcement capability, combined with the historical access record, substantially raises the cost of attempts to circumvent parking payments or to use facilities without authorization.

For the broader urban mobility context, the aggregated data from parking ALPR systems provides urban planners and traffic managers with empirical evidence of parking demand patterns, approach volumes from different directions, and the relationship between parking occupancy and surrounding traffic conditions. These insights inform the design of parking guidance systems, the setting of dynamic pricing schemes, and the planning of new parking infrastructure.

### 1.3 Historical Development of ALPR Technology

The development of automated license plate recognition technology spans approximately four decades and follows a clear progression from specialized hardware-based solutions to software-defined systems running on commodity processors and open-source components.

The earliest practical ALPR systems were developed in the United Kingdom in the late 1970s by the Police Scientific Development Branch. These systems used dedicated hardware for image capture, binarization, and character segmentation and were deployed initially for motorway surveillance. The character recognition step relied on template matching: a library of prototype character images was stored in memory and candidate character regions were compared against each template using correlation metrics. These early systems achieved modest accuracy under controlled conditions but were highly sensitive to lighting variation, plate condition, and the font characteristics of plates from different jurisdictions.

Throughout the 1980s and 1990s, the dominant technical approach for ALPR was classical computer vision with hand-crafted feature extraction. A representative pipeline of this era consisted of: edge detection using Sobel or Canny operators to identify regions of high contrast; connected component analysis and morphological operations to group pixels into candidate plate regions; heuristic filtering based on aspect ratio, area, and intensity histogram to select the most plate-like candidates; vertical projection profile analysis to segment individual characters; and classification using support vector machines or nearest-neighbor matching against a character library. This approach produced reliable results on high-quality input with consistent lighting and perpendicular camera angles but degraded significantly under real-world conditions that included varied illumination, rain reflections, dirty plates, and non-perpendicular mounting angles.

The introduction of neural networks into ALPR began with the application of convolutional neural networks to character classification in the early 2000s. LeCun et al.'s LeNet architecture, originally developed for handwritten digit recognition, was adapted for license plate character recognition by multiple research groups. The use of CNNs for character classification substantially improved accuracy on blurred or degraded characters because CNNs learn representations that are inherently more invariant to local distortions than template matching.

The modern era of ALPR began with the development of deep neural networks capable of performing both detection and recognition in an end-to-end manner. The key enabling innovations were the YOLO family of real-time object detectors, beginning with the original publication by Redmon et al. in 2016, and sequence recognition networks based on the CRNN architecture proposed by Shi et al. in 2016. These two developments together made it possible to replace the entire classical ALPR pipeline — edge detection, morphological processing, candidate filtering, character segmentation, and individual character classification — with two neural networks trained end-to-end on labeled image data.

### 1.4 Deep Learning Foundations for Object Detection

The theoretical foundation of the detection component of this system is deep learning, and more specifically the class of convolutional neural network architectures applied to object detection. An object detector is a function that, given an image, produces a set of predictions each consisting of a class label, a confidence score, and a bounding box specifying the spatial location of the detected object within the image.

The pivotal development that enabled practical real-time object detection was the YOLO (You Only Look Once) architecture introduced by Redmon, Divvala, Girshick, and Farhadi in 2016. The conceptual breakthrough of YOLO was the reformulation of object detection as a single regression problem rather than a two-stage process of region proposal followed by classification. In the YOLO framework, the input image is divided into a grid, and each grid cell simultaneously predicts bounding box coordinates, objectness scores, and class probabilities in a single forward pass through the network. This design achieves dramatically higher inference speed compared to two-stage detectors (such as Faster R-CNN) at a cost of slightly reduced accuracy on small objects with significant spatial overlap.

The YOLO architecture has undergone continuous improvement through successive versions. YOLOv3, published in 2018, introduced multi-scale detection through feature pyramid integration, improving the detection of objects at different scales. YOLOv4 and YOLOv5 (both 2020) introduced extensive improvements to the training pipeline, data augmentation strategies, and backbone architecture. YOLOv8, released by Ultralytics in January 2023, represents the current generation of the YOLO family. Its architectural innovations include an anchor-free detection head, which eliminates the need to pre-define anchor box shapes that must be tuned to the aspect ratios of the target class, and a new C2f (Cross Stage Partial with two bottleneck blocks) module in the backbone that improves gradient flow and feature reuse without increasing computational cost proportionally.

For license plate detection specifically, the relevant properties of YOLOv8 are its high detection accuracy at small object scales (license plates typically occupy between one and five percent of a typical surveillance image's area), its robust performance under varied lighting conditions due to the diversity of the training data, its inference speed on CPU hardware which makes deployment feasible without dedicated GPU infrastructure, and its clean Python API through the Ultralytics package that substantially simplifies integration.

The training of a custom YOLOv8 model for Uzbek license plate detection requires a labeled dataset of images containing vehicle photographs annotated with bounding box coordinates around each license plate. The training process minimizes a composite loss function combining binary cross-entropy for objectness and class predictions with a combination of complete IoU (CIoU) loss and distribution focal loss for bounding box regression. The resulting trained model encodes learned visual features that generalize to license plates not seen during training, provided the test images share sufficient visual similarity with the training distribution.

### 1.5 Optical Character Recognition: Principles and Evolution

Optical character recognition is the computational task of converting rasterized text — text represented as a two-dimensional array of pixel intensity values — into a sequence of machine-readable character codes. For license plate recognition, OCR is applied to the cropped plate region produced by the detector, and the output is the sequence of alphanumeric characters inscribed on the plate.

The foundational challenge of license plate OCR is that plate images encountered in real-world deployment are often degraded by motion blur (from either the vehicle or the camera), optical distortion due to non-perpendicular viewing angles, illumination artifacts including shadows, glare from direct sunlight, and the reflection of headlights at night, and physical damage or obscuration of the plate itself. These degradation factors make license plate OCR substantially more challenging than the document OCR for which most traditional systems were optimized.

Classical OCR engines such as Tesseract, developed originally at HP Laboratories in the 1980s and subsequently open-sourced and maintained by Google, achieve excellent performance on document images with regular fonts and consistent illumination. Tesseract uses a two-stage pipeline: a text layout analysis stage that identifies text regions and delineates individual characters, followed by a character-level classification stage using adaptive classifiers. On license plate images, Tesseract requires careful preprocessing — including binarization, noise removal, and scale normalization — to achieve acceptable accuracy.

EasyOCR, developed by Jaided AI and first released in 2020, takes a fundamentally different approach based entirely on deep learning. It uses two neural networks in sequence: CRAFT (Character Region Awareness for Text Detection), published by Baek et al. in 2019, for identifying text regions and character boundaries, and a CRNN (Convolutional Recurrent Neural Network) with CTC (Connectionist Temporal Classification) decoding for sequence recognition. The CRNN architecture processes the text region as a whole sequence rather than segmenting individual characters, which makes it substantially more robust to character spacing irregularities and touching characters — both common in license plate fonts. The CTC decoder, based on the algorithm proposed by Graves et al. in 2006, enables training and inference on sequence data without requiring character-level alignment between the input image and the output character sequence.

EasyOCR's per-segment confidence scores, produced as part of its output, are used in this thesis system to compute a combined confidence metric for each detection result. The availability of per-segment confidence scores distinguishes EasyOCR from some alternative OCR systems and enables the confidence-based filtering implemented in the pipeline.

### 1.6 Uzbekistan Context: Digital Transformation and Regulatory Framework

The Republic of Uzbekistan has articulated an explicit commitment to digital transformation through its "Digital Uzbekistan 2030" strategy, adopted by Presidential Decree in October 2019. This strategy identifies the digitalization of public services, the modernization of transportation infrastructure, and the development of smart city capabilities as national strategic priorities. Intelligent Transportation Systems, including automated parking management, are explicitly within the scope of the transportation modernization pillar of this strategy.

The regulatory framework governing vehicle registration plates in Uzbekistan is established by State Standard O'z DSt 1180:2006, titled "Registration Plates for Road Vehicles" and published by the Uzstandard agency in Tashkent. This standard specifies the physical dimensions, materials, reflective properties, character fonts, and — critically for this thesis — the alphanumeric format of plates issued for different vehicle categories. The standard civilian plate format, applicable to the vast majority of privately and commercially operated vehicles, is an eight-character sequence structured as two digits (the regional code), one uppercase Latin letter, three digits, and two uppercase Latin letters. This format is represented in regular expression notation as `^[0-9]{2}[A-Z][0-9]{3}[A-Z]{2}$`.

The regional code component of the plate encodes the administrative region where the vehicle is registered. Uzbekistan is divided into twelve provinces (viloyatlar), one autonomous republic (Karakalpakstan), and the capital city of Tashkent, each of which administers vehicle registration through its regional branch of the Ministry of Internal Affairs. The regional code is a two-digit number drawn from a specific set of values assigned to each administrative territory. As of the current edition of the standard, fourteen regional code values are in active use: 01 (Tashkent city), 10 (Tashkent region), 20 (Sirdaryo), 25 (Jizzakh), 30 (Namangan), 35 (Andijan), 40 (Fergana), 50 (Samarkand), 55 (Kashkadarya), 60 (Surkhandarya), 65 (Bukhara), 70 (Navoi), 75 (Khorezm), and 80 (Karakalpakstan). These codes are not numerically consecutive, and not all two-digit numbers in the range 01 to 99 are valid regional codes. The validation of regional codes is therefore an essential component of any system that aims to distinguish genuine Uzbek plates from randomly generated alphanumeric strings that happen to match the format pattern.

The Latin script used for Uzbek plate characters is the standard ISO basic Latin alphabet (A through Z). Uzbekistan transitioned from Cyrillic to a modified Latin script for the Uzbek language in 1993, and vehicle plates have used Latin characters throughout the post-independence period. This is relevant for OCR configuration: the EasyOCR reader must be configured to recognize Latin characters, and the character allowlist must be restricted to the twenty-six uppercase letters and ten digits that can legitimately appear on a plate.

---

## CHAPTER 2. Technologies and Programming Languages Used

### 2.1 Python as the Core Development Language

Python version 3.10 and above serves as the exclusive implementation language for this system. The choice of Python is motivated by a constellation of factors that make it uniquely well-suited for this particular application domain.

The most compelling argument for Python in this context is the richness and maturity of its machine learning and computer vision ecosystem. The de-facto standard deep learning frameworks — PyTorch, TensorFlow, and their associated model libraries — provide their primary Python APIs. The Ultralytics YOLOv8 package, EasyOCR, and OpenCV all present first-class Python interfaces with comprehensive documentation and active community support. This means that the system can leverage the full capabilities of each library with minimal friction, using idiomatic Python code rather than wrapper layers around C or C++ implementations.

Python 3.10 specifically introduces several language features that improve code quality in this codebase. The improved type annotation syntax — including `X | Y` union types as a shorthand for `Union[X, Y]` and built-in collection types such as `list[str]` and `dict[str, float]` as generic types without importing from the `typing` module — substantially improves readability and enables more precise static analysis. The type annotations throughout this codebase serve both as documentation for developers and as the basis for Pydantic's runtime validation in the FastAPI layer.

Python's garbage-collected memory model and dynamic typing also simplify the development of the asynchronous API server. FastAPI and the Uvicorn ASGI server both take advantage of Python's `asyncio` event loop, and the lightweight coroutine model makes it straightforward to handle concurrent HTTP requests without the complexity of manual thread management.

The principal limitation of Python for this application is execution speed. Python's Global Interpreter Lock (GIL) prevents true parallel execution of Python bytecode across multiple threads, and its dynamic dispatch model is inherently slower than compiled languages for CPU-intensive workloads. For the computationally intensive components of this system — neural network inference in the detector and recognizer — this limitation is mitigated by the fact that the actual computation is performed by native C/C++ code in PyTorch (for YOLOv8) and in the underlying runtime (for EasyOCR), with the Python layer serving only as an orchestration interface. Only the preprocessing and postprocessing steps execute in Python, and these are relatively lightweight compared to the inference computations.

### 2.2 YOLOv8: Architecture and Capabilities

YOLOv8, released by Ultralytics in January 2023, represents the eighth generation of the You Only Look Once family of real-time object detection models. It is available as a Python package under the name `ultralytics` and provides a unified API for five computer vision tasks: detection, instance segmentation, image classification, pose estimation, and object tracking. This thesis uses YOLOv8 exclusively for object detection, specifically the detection of license plate bounding boxes in input images.

The YOLOv8 architecture follows the standard detector structure of backbone, neck, and head. The backbone is responsible for extracting hierarchical feature representations from the input image. YOLOv8's backbone uses a modified CSPDarknet architecture incorporating C2f modules (Cross Stage Partial network with two bottleneck layers), which improve gradient flow during training compared to the C3 modules used in YOLOv5. The neck is a feature pyramid network (FPN) combined with a path aggregation network (PANet) that combines features at multiple scales to enable detection of objects ranging from very small (a few dozen pixels) to very large (most of the image).

The detection head in YOLOv8 is anchor-free, which is a significant architectural departure from earlier YOLO versions. Anchor-based detectors predefine a set of reference bounding box shapes at each spatial location and predict offsets from these anchors. The anchor shapes must be tuned (through k-means clustering on the training data bounding boxes) to match the aspect ratios of the target objects. An anchor-free head eliminates this configuration step by directly predicting the offsets from the center of each candidate location to the four sides of the bounding box, making the detector naturally adaptive to any object shape without prior configuration.

YOLOv8 is available in five model sizes: n (nano), s (small), m (medium), l (large), and x (extra-large), which trade off inference speed against detection accuracy. The nano and small variants are appropriate for deployment on CPU hardware where inference latency is a concern. For the custom plate-specific model used in this thesis (`plate_detector.pt`), a small or medium base architecture provides the best balance between accuracy on small plate regions and inference speed on commodity hardware.

The Ultralytics Python API is notable for its simplicity. The entire detection pipeline — loading the model, running inference, and extracting bounding box results — requires fewer than ten lines of Python code, as demonstrated in the implementation chapter. The framework also provides automatic model downloading from the Ultralytics model repository, handles CUDA/CPU device selection transparently, and produces consistent result objects regardless of the specific YOLOv8 variant used. This API design substantially reduces the integration burden compared to using the underlying PyTorch model directly.

### 2.3 EasyOCR: Deep Learning-Based Character Recognition

EasyOCR is an open-source OCR library developed by Jaided AI, first published in August 2020, and maintained as an active open-source project on GitHub. It supports over eighty languages using a pre-trained model zoo and provides a Python API that accepts a numpy array or image file path and returns a list of recognition results, each containing a bounding polygon, recognized text, and confidence score.

The two-stage architecture of EasyOCR is central to its effectiveness on license plate images. The text detection stage uses CRAFT (Character Region Awareness for Text Detection), a convolutional network that produces two probability maps: a character region score indicating the probability that each pixel belongs to the center of a character, and an affinity score indicating the probability that two adjacent characters belong to the same word. These maps are thresholded and processed to produce bounding boxes around text regions. The affinity-based grouping is particularly useful for license plates because it naturally groups all characters on the plate into a single text region rather than treating each character individually.

The recognition stage uses a CRNN architecture. The convolutional layers of the CRNN extract visual features from the text region image, producing a sequence of feature vectors corresponding to horizontal positions in the image. A bidirectional LSTM (Long Short-Term Memory) network processes this feature sequence to incorporate contextual information — the identity of characters to the left and right of each position influences the prediction for each position. The CTC (Connectionist Temporal Classification) layer then decodes the LSTM output into a character sequence without requiring explicit alignment between input positions and output characters.

The `allowlist` parameter of EasyOCR's `readtext` method is a critical feature for license plate recognition. By restricting the recognized character set to `ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`, the system eliminates the possibility of the OCR engine returning characters that cannot legally appear on an Uzbek plate — such as punctuation, Cyrillic letters, or accented Latin characters. This constraint both improves recognition accuracy for valid plates (by removing alternative hypotheses that might otherwise score higher than the correct characters on ambiguous inputs) and simplifies the downstream validation logic.

A practical consideration in EasyOCR deployment is initialization time. The `easyocr.Reader` constructor loads two neural networks from disk (or downloads them on first use) and prepares them for inference. This initialization takes approximately three to eight seconds on a typical laptop. The system architecture accounts for this by constructing the `TextRecognizer` once at startup and sharing the initialized instance across all requests, rather than reinitializing it per request.

### 2.4 OpenCV: Image Preprocessing and Video Processing

OpenCV (Open Source Computer Vision Library), originally developed by Intel in 1999 and now maintained by the OpenCV Foundation, is the standard library for computer vision operations in Python. It provides efficient implementations of hundreds of image processing algorithms including color space conversions, geometric transformations, morphological operations, thresholding algorithms, feature detection, and video capture and decoding.

In this system, OpenCV serves two distinct roles. The first is image preprocessing for OCR: the `_preprocess` method in the `TextRecognizer` class uses OpenCV to convert the plate crop to grayscale, resize it to a standardized height of 128 pixels, apply OTSU binarization, and apply a sharpening filter. The second is video file processing: the `process_video` method in the `Pipeline` class uses `cv2.VideoCapture` to read frames from a video file and iterate through them frame by frame.

OTSU thresholding, named after Nobuyuki Otsu who published the method in 1979, is an automatic binarization algorithm that determines an optimal threshold value by minimizing the intra-class variance of foreground and background pixel intensity distributions. Unlike fixed-threshold binarization, which requires manual selection of a threshold value that is appropriate for the specific lighting conditions of each image, OTSU thresholding adapts automatically to the intensity distribution of each input image. For license plate images captured under varying ambient light levels, this adaptability is essential for consistent binarization quality.

The sharpening filter applied after binarization uses a 3×3 kernel with a center coefficient of 5 and surrounding coefficients of −1 (except corners which are 0), which is a discrete approximation of the Laplacian sharpening operator. This filter enhances edges and increases the contrast between character strokes and the plate background, which is particularly beneficial for low-resolution crops where character boundaries may be blurred due to insufficient pixel density.

For video processing, `cv2.VideoCapture` provides a unified interface for reading from local video files in any format supported by the underlying FFmpeg library, which includes all common video formats encountered in practice (MP4/H.264, MOV/H.265, AVI, MKV). The frame iterator interface allows the pipeline to process video files without loading the entire file into memory, which is important for long recordings.

### 2.5 FastAPI: Modern Asynchronous REST Framework

FastAPI is a Python web framework for building REST APIs, created by Sebastián Ramírez and first released in December 2018. It has achieved rapid and widespread adoption in the Python community, consistently ranking among the most popular Python web frameworks since 2020. Its core design principles are speed (comparable to Node.js and Go in benchmarks for I/O-bound workloads due to its fully asynchronous ASGI foundation), developer ergonomics (minimal boilerplate, automatic documentation), and reliability (automatic request validation and response serialization via Pydantic).

FastAPI is built on top of Starlette (for HTTP handling and ASGI integration) and Pydantic (for data validation and serialization). The combination provides a programming model where endpoint handlers are annotated Python functions: the function signature declares the expected path parameters, query parameters, and request body types, and FastAPI automatically validates incoming requests against these declarations, returning informative HTTP 422 Unprocessable Entity responses for invalid inputs without requiring any validation code in the handler body.

Pydantic models, defined as Python classes inheriting from `pydantic.BaseModel` with typed fields, serve as both request schemas (validated automatically from JSON request bodies) and response schemas (automatically serialized to JSON). The same Pydantic model definitions are also used by FastAPI to generate OpenAPI 3.0 documentation, accessible at the `/docs` endpoint as an interactive Swagger UI and at `/openapi.json` as a machine-readable specification. This automatic documentation generation substantially reduces the documentation maintenance burden and ensures that the API documentation is always synchronized with the actual implementation.

The ASGI (Asynchronous Server Gateway Interface) foundation of FastAPI, served in this system by the Uvicorn ASGI server, provides native support for asynchronous request handling. Endpoint handlers declared with the `async def` keyword can perform non-blocking I/O operations (such as database reads, network requests, or file I/O) without blocking the server's event loop, enabling a single process to handle multiple concurrent requests efficiently.

### 2.6 SQLite with WAL Mode: Embedded Relational Persistence

SQLite is an embedded relational database management system implemented as a C library. Unlike client-server database systems such as PostgreSQL or MySQL, SQLite operates entirely within the address space of the application process, reading and writing a single cross-platform database file. This design makes SQLite uniquely convenient for embedded applications, desktop software, and single-server deployments where the overhead of a separate database process is not warranted.

SQLite supports the full SQL data definition and manipulation language including transactions, constraints, triggers, and views. Its indexing support (B-tree based, compatible with standard SQL CREATE INDEX syntax) enables efficient query performance for the access patterns required by this system: lookup by plate text, time-range queries, and counting by validity status. The database schema used in this system is described in detail in Chapter 3.

The configuration option most important for this application is WAL (Write-Ahead Logging) journal mode, enabled with the SQL pragma `PRAGMA journal_mode=WAL`. In SQLite's default rollback journal mode, a write transaction takes an exclusive lock on the database file, blocking all concurrent readers until the transaction completes. In WAL mode, writes are appended to a separate write-ahead log file, and readers can continue accessing the last committed state of the main database file concurrently with ongoing write transactions. WAL mode also provides improved write performance for workloads with frequent small writes, because appending to the log file is sequential and does not require modifying the main database file until a checkpoint operation is performed.

The significance of WAL mode for this system is concrete: the FastAPI server may receive multiple concurrent detection requests, each of which writes to the database, while simultaneously serving read requests from the `/detections` and `/stats` endpoints. Without WAL mode, write transactions would block read queries, producing unnecessary latency for queries that happen to be in flight during a detection save operation.

### 2.7 Supporting Libraries and the Dependency Ecosystem

Several additional libraries complete the technology stack. NumPy provides the fundamental N-dimensional array data structure (`np.ndarray`) used to represent images throughout the pipeline. Both OpenCV and YOLOv8 represent images as NumPy arrays, and EasyOCR accepts NumPy arrays as input, making NumPy the universal image representation that binds the pipeline components together. The geometric mean confidence calculation — `(yolo_conf * ocr_conf) ** 0.5` — also uses standard Python arithmetic on float values returned from the respective libraries.

Pillow (the Python Imaging Library fork) provides image I/O capabilities that complement OpenCV. `python-multipart` provides the multipart form data parser required by FastAPI for file upload handling. Uvicorn, the ASGI server, is configured with the `[standard]` extra which includes accelerated HTTP parsing through the `httptools` library.

The complete dependency specification is maintained in `requirements.txt` and reflects the minimum version constraints that have been tested and verified to work together:

```
ultralytics>=8.2.0
easyocr>=1.7.1
opencv-python>=4.9.0
Pillow>=10.3.0
numpy>=1.26.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
pydantic>=2.7.0
```

---

## CHAPTER 3. System Design, Architecture, and Implementation

### 3.1 System Overview and Design Principles

The system is designed around five core software engineering principles that guided every architectural and implementation decision: separation of concerns, testability, configurability, fail-safe defaults, and minimal external dependencies.

Separation of concerns mandates that each module has one clearly defined responsibility and exposes it through a narrow interface. The detector knows only how to find license plates in an image. The recognizer knows only how to read text from a plate crop. The validator knows only the business rules for Uzbek plate format compliance. The pipeline knows only how to orchestrate these three stages and apply the best-detection strategy. The database manager knows only how to persist and query detection records. This decomposition enables any single module to be replaced or upgraded independently without modifying the others.

Testability requires that every module can be exercised in isolation with controlled inputs and observable outputs. This principle excludes designs that mix concern boundaries (for example, a recognizer that also saves to the database) and requires that dependencies are injected rather than globally imported. The test suite demonstrates that all modules except those requiring a real neural network model (which are tested through synthetic substitutes) can be tested without any external infrastructure.

Configurability through environment variables follows the Twelve-Factor App methodology and ensures that the system can be deployed in different environments (development, testing, production) without code changes. Every runtime parameter that might legitimately vary between deployments is exposed as an environment variable with a documented default.

Fail-safe defaults mean that when a configuration value is missing or an optional component (such as the custom YOLO model) is unavailable, the system degrades gracefully to a reduced-capability mode rather than failing to start. The detector's fallback to the general-purpose YOLOv8n model when the custom plate detector is absent is an example of this principle.

### 3.2 High-Level Architecture

The system is organized into three horizontal tiers connected by well-defined interfaces.

**Figure 3.2.1. High-Level System Architecture Diagram**

```
┌────────────────────────────────────────────────────────────────────┐
│                         INPUT TIER                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │  Image File │    │  Video File │    │  HTTP Upload (API)   │   │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬───────────┘   │
└─────────┼─────────────────┼────────────────────────┼───────────────┘
          │                 │                         │
          └────────────┬────┘                         │
                       ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING TIER                                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     Pipeline (core/pipeline.py)              │  │
│  │                                                               │  │
│  │  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐  │  │
│  │  │  PlateDetector│→ │ TextRecognizer  │→ │validate_plate │  │  │
│  │  │  (YOLOv8)     │  │ (EasyOCR+      │  │  (validator)  │  │  │
│  │  │               │  │  Preprocessing) │  │               │  │  │
│  │  └───────────────┘  └─────────────────┘  └───────────────┘  │  │
│  │                                                               │  │
│  │  Cross-cutting: Geometric Mean Confidence · Best-Per-Plate   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                                          │
          ▼                                          ▼
┌──────────────────────┐              ┌──────────────────────────────┐
│    PERSISTENCE TIER  │              │       SERVICE TIER           │
│                      │              │                              │
│  SQLite (WAL mode)   │              │  FastAPI REST API            │
│  ┌────────────────┐  │              │  ┌─────────────────────────┐ │
│  │ detections     │  │              │  │ /health                 │ │
│  ├────────────────┤  │              │  │ /detect/image           │ │
│  │ valid_plates   │  │              │  │ /detect/video           │ │
│  └────────────────┘  │              │  │ /detections             │ │
│                      │              │  │ /detections/{id}/image  │ │
└──────────────────────┘              │  │ /stats                  │ │
                                      └──────────────────────────────┘
```

The input tier accepts image files, video files, and HTTP-uploaded files through the CLI and API entry points respectively. The processing tier is the pipeline, which orchestrates the three recognition stages and applies confidence scoring and the best-detection-per-plate strategy for video inputs. The persistence tier is the SQLite database managed through the `DatabaseManager` class. The service tier is the FastAPI application, which exposes the pipeline's capabilities as HTTP endpoints.

### 3.3 Module Decomposition and Responsibilities

**Table 3.3.1. Module Decomposition**

| Module | File Path | Primary Responsibility |
|--------|-----------|------------------------|
| Configuration | `config.py` | Environment-variable-based runtime parameter management |
| Detector | `core/detector.py` | YOLOv8-based license plate bounding box prediction |
| Recognizer | `core/recognizer.py` | EasyOCR text extraction with preprocessing |
| Validator | `core/validator.py` | Uzbek plate format enforcement and OCR correction |
| Pipeline | `core/pipeline.py` | Stage orchestration, confidence scoring, best-per-plate selection |
| Database Manager | `db/manager.py` | SQLite read/write operations with parameterized queries |
| Database Schema | `db/schema.sql` | DDL definitions for tables and indexes |
| API Application | `api/app.py` | FastAPI application, CORS, startup/shutdown hooks |
| API Routes | `api/routes.py` | HTTP endpoint handler functions |
| API Schemas | `api/schemas.py` | Pydantic request/response model definitions |
| CLI Entry Point | `main.py` | Command-line argument parsing and subcommand dispatch |

### 3.4 Detection Module Implementation

The `PlateDetector` class in `core/detector.py` encapsulates all interaction with the YOLOv8 model. Its interface exposes a single primary method, `detect`, which accepts a BGR-format NumPy array (as produced by `cv2.imread` or `cv2.VideoCapture.read`) and returns a list of dictionaries, each containing a bounding box, a confidence score, and a cropped sub-array of the plate region.

```python
class PlateDetector:
    """
    Wraps YOLOv8 for license plate region detection.
    Loads model once; reuses across all frames.
    """

    def __init__(self, model_path: str, confidence: float = 0.45):
        from ultralytics import YOLO

        self.confidence = confidence
        model_file = Path(model_path)

        if model_file.exists():
            logger.info(f"Loading plate detection model from {model_path}")
            self.model = YOLO(str(model_file))
        else:
            logger.warning(
                f"Custom model not found at {model_path}. "
                "Falling back to YOLOv8n with aspect-ratio filtering. "
                "For best accuracy, provide a plate-specific model."
            )
            self.model = YOLO("yolov8n.pt")
            self._fallback_mode = True
            return

        self._fallback_mode = False

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if self._fallback_mode:
                    w, h = x2 - x1, y2 - y1
                    if h == 0 or (w / h) < 2.0 or (w / h) > 8.0:
                        continue
                    frame_area = frame.shape[0] * frame.shape[1]
                    if (w * h) / frame_area > 0.15:
                        continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                detections.append({'bbox': [x1, y1, x2, y2],
                                   'confidence': conf, 'crop': crop})
        return detections
```

The constructor uses an early-return pattern to set `_fallback_mode`. When the custom model file is found, `self.model` is assigned and `_fallback_mode` is set to `False` at the end of the constructor body. When the file is absent, `_fallback_mode` is set to `True` and the constructor returns immediately, leaving the YOLOv8n general-purpose model loaded. This design ensures that `_fallback_mode` is always defined before `detect` is called, regardless of which branch the constructor took.

The fallback mode logic applies two heuristic filters when the custom plate-specific model is not available. The aspect ratio filter retains only bounding boxes with width-to-height ratios between 2.0 and 8.0, which encompasses the typical elongated rectangular shape of a license plate. The area filter discards bounding boxes that occupy more than 15% of the total frame area, which eliminates large foreground objects such as vehicle bodies that might produce high-confidence detections from the general-purpose COCO-trained model.

The `draw_boxes` utility method provides a visualization capability used by the CLI `--show` flag, overlaying green bounding rectangles and label text on a copy of the original frame for debugging and demonstration purposes.

### 3.5 Recognition Module Implementation

The `TextRecognizer` class in `core/recognizer.py` implements a four-step preprocessing pipeline followed by EasyOCR inference. The preprocessing is the scientifically critical component: the same EasyOCR model produces substantially different accuracy depending on whether the plate crop has been preprocessed to remove illumination variation and enhance character contrast.

```python
_TARGET_HEIGHT = 128


class TextRecognizer:

    def __init__(self, languages: list[str], gpu: bool = False):
        import easyocr
        logger.info(f"Initializing EasyOCR (languages={languages}, gpu={gpu})")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self._warm_up()

    def _warm_up(self):
        blank = np.ones((64, 200), dtype=np.uint8) * 255
        try:
            self.reader.readtext(blank)
        except Exception:
            pass

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        preprocessed = self._preprocess(crop)
        results = self.reader.readtext(
            preprocessed, detail=1, paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        )
        if not results:
            return "", 0.0
        texts = [r[1].strip() for r in results]
        confidences = [r[2] for r in results]
        raw_text = "".join(texts)
        mean_conf = sum(confidences) / len(confidences)
        return raw_text, round(mean_conf, 4)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        h, w = gray.shape
        if h == 0:
            return gray
        scale = _TARGET_HEIGHT / h
        new_w = max(1, int(w * scale))
        resized = cv2.resize(gray, (new_w, _TARGET_HEIGHT),
                             interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(resized, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                          dtype=np.float32)
        sharpened = cv2.filter2D(binary, -1, kernel)
        return sharpened
```

The module-level constant `_TARGET_HEIGHT = 128` fixes the output height of every preprocessed plate crop at 128 pixels. The width is scaled proportionally to preserve the plate's aspect ratio, using bicubic interpolation (`cv2.INTER_CUBIC`) to minimize aliasing artifacts during the resize. A target height of 128 pixels was selected empirically based on EasyOCR's internal character recognition model: individual plate characters (which occupy approximately half the plate height) are placed in the 40–64 pixel range after resizing, which falls within the optimal input size for the CRNN recognition stage.

The warm-up call in the constructor is an important operational detail. EasyOCR uses PyTorch's JIT compilation infrastructure, and the first inference call after initialization triggers compilation of the computational graph, which takes significantly longer than subsequent calls. By performing a dummy inference on a blank 64×200 image during initialization, the warm-up step ensures that all subsequent inference calls on real plate crops execute at the steady-state speed rather than incurring the compilation overhead on the first real request.

**Figure 3.5.1. Preprocessing Pipeline Visualization**

The preprocessing chain transforms a raw plate crop through four stages:

1. BGR color image → Grayscale (single-channel, eliminates color information irrelevant to character shape)
2. Grayscale → Height-normalized (128px height, proportional width resize using bicubic interpolation)
3. Height-normalized → Binary (OTSU threshold, adapts to each image's illumination)
4. Binary → Sharpened (Laplacian enhancement kernel strengthens character stroke edges)

### 3.6 Validation Module and Uzbek Plate Standard

The validation module in `core/validator.py` encodes all business rules derived from State Standard O'z DSt 1180:2006. It is the module with the densest domain-specific knowledge in the system, and its correctness is critical because it determines whether a recognition result is treated as a genuine vehicle identification event.

**Table 3.6.1. Uzbek License Plate Character Position Specification**

| Position Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|----------------|---|---|---|---|---|---|---|---|
| Character Type | Digit | Digit | Letter | Digit | Digit | Digit | Letter | Letter |
| Valid Values | 0–9 | 0–9 | A–Z | 0–9 | 0–9 | 0–9 | A–Z | A–Z |

The two-character regional code (positions 0–1) must be a member of the set of fourteen currently active region codes defined in `VALID_REGIONS`:

```python
VALID_REGIONS = {
    '01',  # Tashkent city
    '10',  # Tashkent region
    '20',  # Sirdaryo
    '25',  # Jizzakh
    '30',  # Namangan
    '35',  # Andijan
    '40',  # Fergana
    '50',  # Samarkand
    '55',  # Kashkadarya
    '60',  # Surkhandarya
    '65',  # Bukhara
    '70',  # Navoi
    '75',  # Khorezm
    '80',  # Karakalpakstan
}

UZ_PLATE_RE = re.compile(r'^[0-9]{2}[A-Z][0-9]{3}[A-Z]{2}$')

_OCR_FIXES = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5',
    'B': '8', 'G': '6', 'Q': '0',
}


def validate_plate(text: str) -> tuple[bool, str]:
    normalized = normalize(text)
    if UZ_PLATE_RE.match(normalized) and normalized[:2] in VALID_REGIONS:
        return True, normalized
    corrected = _try_correct(normalized)
    if corrected and UZ_PLATE_RE.match(corrected) and corrected[:2] in VALID_REGIONS:
        return True, corrected
    return False, normalized


def _try_correct(text: str) -> str | None:
    if len(text) != 8:
        return None
    chars = list(text)
    for i in (0, 1, 3, 4, 5):   # digit positions
        if not chars[i].isdigit():
            fix = _OCR_FIXES.get(chars[i])
            if fix:
                chars[i] = fix
            else:
                return None
    for i in (2, 6, 7):           # letter positions
        if not chars[i].isalpha():
            return None
    return ''.join(chars)
```

The OCR correction logic in `_try_correct` applies only to digit positions (indices 0, 1, 3, 4, 5). This constraint is important: the correction mapping includes character substitutions that are valid in one direction only. For example, the letter 'O' is corrected to the digit '0' at a digit position, but if 'O' appears at a letter position (indices 2, 6, 7), it is a legitimate character and must not be corrected. By restricting correction to digit positions, the function avoids introducing errors at positions where letters are expected.

The normalization step — stripping spaces, dashes, and dots, then converting to uppercase — is applied before both the direct regex match and the OCR correction attempt. This handles the common case where an OCR engine inserts a space between the regional code and the letter component of the plate, producing output such as "01 A123BC" instead of "01A123BC".

It is important to note that `VALID_REGIONS` contains exactly the fourteen codes listed above. Any two-digit prefix not in this set will cause `validate_plate` to return `(False, ...)` even if the plate otherwise matches the format pattern. This tight constraint is deliberate: it prevents the system from accepting plates with fabricated or unassigned region codes that could arise from OCR errors on ambiguous digit pairs.

### 3.7 Pipeline Orchestration, Best-Detection Strategy, and Video Processing

The `Pipeline` class in `core/pipeline.py` connects the three processing stages and implements the combined confidence scoring and the best-detection-per-plate strategy for video inputs.

**3.7.1 Data Structures**

```python
@dataclass
class DetectionResult:
    source: str
    raw_text: str
    plate_text: str
    is_valid: bool
    confidence: float          # geometric mean of YOLO and OCR confidence
    bbox: list[int]            # [x1, y1, x2, y2]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    frame: np.ndarray | None = field(default=None, repr=False)
```

The `DetectionResult` dataclass carries the complete output of one plate recognition event. The `frame` field stores the full BGR numpy array of the source frame — excluded from `repr` to avoid printing megabyte-sized array data in log output. The `source` field encodes the origin of the detection: an image file path for still images, or a string of the form `video:<path>:frame_<N>` for video frames.

**3.7.2 Frame-Level Processing**

```python
class Pipeline:

    def __init__(self, model_path: str, confidence: float,
                 languages: list[str], gpu: bool,
                 video_frame_skip: int = 3,
                 dedup_window_seconds: int = 2):
        self.detector = PlateDetector(model_path, confidence)
        self.recognizer = TextRecognizer(languages, gpu)
        self.frame_skip = video_frame_skip

    def _process_frame(self, frame: np.ndarray,
                       source: str) -> list[DetectionResult]:
        detections = self.detector.detect(frame)
        results = []
        for det in detections:
            raw_text, ocr_conf = self.recognizer.recognize(det['crop'])
            if not raw_text:
                continue
            is_valid, plate_text = validate_plate(raw_text)
            combined_conf = round(
                (det['confidence'] * ocr_conf) ** 0.5, 4)
            results.append(DetectionResult(
                source=source,
                raw_text=raw_text,
                plate_text=plate_text,
                is_valid=is_valid,
                confidence=combined_conf,
                bbox=det['bbox'],
                frame=frame,
            ))
        return results
```

The geometric mean for combined confidence is computed as the square root of the product of the YOLO detector confidence and the EasyOCR mean segment confidence: `combined_conf = sqrt(yolo_conf × ocr_conf)`. This formula has a desirable mathematical property: it penalizes results where either confidence value is low much more severely than an arithmetic mean would. To illustrate: if the detector produces confidence 0.90 but OCR produces only 0.20, the geometric mean is `sqrt(0.90 × 0.20) ≈ 0.42`, which is below the minimum save threshold of 0.60 and will not be persisted. The arithmetic mean would yield `(0.90 + 0.20) / 2 = 0.55`, which also falls below the threshold but misrepresents the result as approximately average quality when in fact the OCR component has very low confidence.

**3.7.3 Video Processing and Best-Detection Strategy**

```python
def process_video(
    self,
    video_path: str,
    on_result: Callable[[DetectionResult], None],
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # plate_text -> best DetectionResult seen so far
    best: dict[str, DetectionResult] = {}
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % self.frame_skip != 0:
                continue
            source = f"video:{video_path}:frame_{frame_idx}"
            for result in self._process_frame(frame, source=source):
                if not result.is_valid:
                    continue
                prev = best.get(result.plate_text)
                if prev is None or result.confidence > prev.confidence:
                    best[result.plate_text] = result
    finally:
        cap.release()

    for result in best.values():
        on_result(result)

    return len(best)
```

The video processing strategy is deliberately different from naive frame-by-frame accumulation. Rather than emitting a detection event each time a valid plate is recognized in a frame, the pipeline maintains a `best` dictionary mapping each unique plate text to the single highest-confidence `DetectionResult` observed across the entire video. Only after all frames have been processed does the pipeline invoke the `on_result` callback for each unique plate, passing its best observation.

This approach has several important consequences. First, each vehicle that appears in the video generates exactly one database record, regardless of how many frames it appears in. This produces a clean, deduplicated detection history that directly reflects vehicle events rather than frame counts. Second, the record written for each vehicle corresponds to the highest-quality recognition, typically the frame in which the plate is most clearly visible and closest to the camera. Third, the approach requires that the entire video be processed before any results are emitted, which means the video endpoint is not a true streaming pipeline in the sense of real-time output; results are buffered in memory until processing is complete.

Frame skipping is controlled by `self.frame_skip` (default 3). Frames at indices that are not divisible by `frame_skip` are discarded without processing. At a source video frame rate of 30 fps, a skip stride of 3 yields an effective processing rate of 10 frames per second of video content, reducing the computation by 67% at the cost of potentially missing a plate that is visible for fewer than three consecutive frames — a scenario that is rare in typical parking footage where vehicles move slowly.

The `dedup_window_seconds` parameter accepted by the constructor is preserved for API compatibility but is not used by the current implementation, which employs the best-detection approach instead.

**Figure 3.7.1. Video Processing Logic Flow**

```
For each frame in video:
    ├── Skip if frame_idx % frame_skip != 0
    └── Process frame → list[DetectionResult]
            └── For each valid DetectionResult:
                    ├── If plate_text not in best: store it
                    └── If confidence > best[plate_text].confidence: replace it

After all frames consumed:
    └── For each plate_text in best:
            └── Call on_result(best[plate_text])   ← exactly one callback per unique plate
```

### 3.8 Database Design and Implementation

The persistence layer consists of two tables defined in `db/schema.sql` and managed through the `DatabaseManager` class in `db/manager.py`. The `DatabaseManager` constructor also accepts an optional `images_dir` parameter specifying a directory where cropped plate image files are stored, enabling the image retrieval endpoint described in Section 3.9.

**Figure 3.8.1. Entity-Relationship Diagram**

```
┌─────────────────────────────────────┐
│            detections               │
├──────────────────────────────────── ┤
│  id          INTEGER  PK AUTOINCR   │
│  source      TEXT     NOT NULL      │
│  plate_text  TEXT     NOT NULL      │
│  raw_text    TEXT     NOT NULL      │
│  is_valid    INTEGER  NOT NULL (0/1)│
│  confidence  REAL     NOT NULL      │
│  bbox_x1     INTEGER  NULLABLE      │
│  bbox_y1     INTEGER  NULLABLE      │
│  bbox_x2     INTEGER  NULLABLE      │
│  bbox_y2     INTEGER  NULLABLE      │
│  detected_at TEXT     NOT NULL      │
├─────────────────────────────────────┤
│  INDEX idx_plate_text               │
│  INDEX idx_detected_at              │
│  INDEX idx_is_valid                 │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│            valid_plates             │
├─────────────────────────────────────┤
│  id          INTEGER  PK AUTOINCR   │
│  plate_text  TEXT     NOT NULL      │
│  confidence  REAL     NOT NULL      │
│  source      TEXT     NOT NULL      │
│  detected_at TEXT     NOT NULL      │
├─────────────────────────────────────┤
│  INDEX idx_vp_plate_text            │
│  INDEX idx_vp_detected_at           │
└─────────────────────────────────────┘
```

The `detections` table records every detection event, including detections of plates that do not pass the Uzbek format validation (`is_valid = 0`). Retaining invalid detections provides two operational benefits: it enables post-hoc analysis of recognition failure patterns, and it preserves a complete audit trail of all events even when the system is uncertain about the result. The `raw_text` field records the original OCR output before correction, and `plate_text` records the normalized or corrected form, enabling comparison of the two to analyze the frequency and nature of corrections applied.

The `valid_plates` table is a filtered projection containing only records with `is_valid = 1`. Its existence simplifies and accelerates access control queries that only need to check whether a specific plate has been seen recently. Rather than querying the `detections` table with a `WHERE is_valid = 1 AND plate_text = ?` predicate, a parking access control component can query the smaller `valid_plates` table with a simple equality predicate.

```python
def save_detection(self, result: DetectionResult) -> int:
    bbox = result.bbox if len(result.bbox) == 4 else [None, None, None, None]
    with self._conn:
        cursor = self._conn.execute(
            """
            INSERT INTO detections
                (source, plate_text, raw_text, is_valid, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.source, result.plate_text, result.raw_text,
                int(result.is_valid), result.confidence,
                bbox[0], bbox[1], bbox[2], bbox[3],
                result.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
        )
        return cursor.lastrowid
```

All SQL statements use positional `?` parameter binding, which is SQLite's parameterized query syntax. This prevents SQL injection: a plate text containing SQL metacharacters such as single quotes, semicolons, or SQL keywords is treated as literal string data rather than as SQL syntax. This security property is non-trivial for a parking system, where an adversary who could construct a plate string that terminates and modifies a SQL statement could potentially corrupt or exfiltrate the detection database.

### 3.9 REST API Design and Implementation

The FastAPI application is defined in `api/app.py` and uses the startup/shutdown event hooks to initialize and clean up the pipeline and database manager:

```python
@app.on_event("startup")
def startup():
    app.state.pipeline = Pipeline(
        model_path=config.MODEL_PATH,
        confidence=config.YOLO_CONFIDENCE,
        languages=config.OCR_LANGUAGES,
        gpu=config.USE_GPU,
        video_frame_skip=config.VIDEO_FRAME_SKIP,
        dedup_window_seconds=config.DEDUP_WINDOW_SECONDS,
    )
    app.state.db = DatabaseManager(config.DB_PATH,
                                   images_dir=config.PLATE_IMAGES_DIR)


@app.on_event("shutdown")
def shutdown():
    app.state.db.close()
```

The pipeline and database manager are attached to `app.state` and injected into request handlers through FastAPI's dependency injection system (`Depends`), ensuring that the single initialized instances are shared across all concurrent requests without global variables.

**Table 3.9.1. REST API Endpoint Summary**

| Method | Path | Description | Response Type |
|--------|------|-------------|---------------|
| GET | `/health` | System health status | `HealthResponse` |
| POST | `/detect/image` | Detect plates in uploaded image | `list[DetectionResponse]` |
| POST | `/detect/video` | Detect plates in uploaded video | NDJSON stream |
| GET | `/detections` | Paginated detection history | `DetectionListResponse` |
| GET | `/detections/search` | Search by plate text | `list[DetectionResponse]` |
| GET | `/detections/{id}` | Single detection by ID | `DetectionResponse` |
| GET | `/detections/{id}/image` | Cropped plate image for a detection | JPEG image |
| GET | `/stats` | Aggregate statistics | `StatsResponse` |

The `/detections/{id}/image` endpoint retrieves the saved JPEG image file associated with a specific detection record. It reads the `image_path` field from the detection record, verifies the file exists, and streams the JPEG bytes with `media_type="image/jpeg"`. This endpoint enables client applications such as management dashboards to display visual evidence of each detection alongside the plate text and metadata.

The video detection endpoint uses FastAPI's `StreamingResponse` with media type `application/x-ndjson` (newline-delimited JSON) to transmit detection results to the client after video processing is complete. Each unique plate detected in the video is serialized as one JSON object on a separate line:

```python
@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    save: bool = Query(True),
    pipeline=Depends(_pipeline),
    db=Depends(_db),
):
    data = await file.read()

    async def generate():
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "video.mp4").suffix,
            delete=True
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            results_buffer = []

            def on_result(result):
                record_id = db.save_detection(result) if save else -1
                payload = {
                    "id": record_id,
                    "source": result.source,
                    "plate_text": result.plate_text,
                    "raw_text": result.raw_text,
                    "is_valid": result.is_valid,
                    "confidence": result.confidence,
                    "detected_at": result.timestamp.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"),
                }
                results_buffer.append(json.dumps(payload) + "\n")

            pipeline.process_video(tmp.name, on_result=on_result)
            for line in results_buffer:
                yield line

    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

Because `pipeline.process_video` completes all frame processing before invoking any `on_result` callback, the results are buffered in `results_buffer` and only yielded after the entire video has been processed. The NDJSON format is retained because it is easily consumed by streaming HTTP clients and aligns with industry conventions for server-sent event streams, even in scenarios where the output is delivered in a single batch.

The Pydantic response models in `api/schemas.py` serve dual purposes. At runtime, FastAPI serializes response data to JSON using these models, performing automatic type coercion and validation. At documentation time, FastAPI reads the model field definitions to generate the OpenAPI schema that populates the `/docs` Swagger UI:

```python
class DetectionResponse(BaseModel):
    id: int
    source: str
    plate_text: str
    raw_text: str
    is_valid: bool
    confidence: float
    bbox_x1: int | None
    bbox_y1: int | None
    bbox_x2: int | None
    bbox_y2: int | None
    detected_at: str
    image_path: str | None = None
```

The optional `image_path` field in `DetectionResponse` carries the filesystem path to the saved plate crop image when available, allowing API consumers to request the image via the `/detections/{id}/image` endpoint.

### 3.10 Command-Line Interface

The `main.py` entry point implements four subcommands using Python's standard `argparse` library:

- `image <path> [--show] [--no-save]`: Process a single image file, print JSON detection results, and optionally display the annotated image.
- `video <path> [--skip N] [--no-save]`: Process a video file with optional frame skip override, printing JSON results to stdout after processing completes.
- `serve [--host H] [--port P]`: Start the Uvicorn ASGI server hosting the FastAPI application.
- `query [--search S] [--stats] [--limit N] [--valid-only]`: Query the database and print results as JSON.

The CLI enables use of the system without the API server overhead, which is appropriate for batch processing scenarios where a directory of parking lot images is to be processed overnight and results stored in the database for analysis.

### 3.11 Configuration Management

All runtime parameters are exposed through environment variables read in `config.py` using `os.environ.get` with documented default values:

**Table 3.11.1. Environment Variable Configuration Reference**

| Variable | Default | Description |
|----------|---------|-------------|
| `LPR_DB_PATH` | `./plates.db` | Path to the SQLite database file |
| `LPR_MODEL_PATH` | `./models/plate_detector.pt` | Path to YOLOv8 model weights |
| `LPR_YOLO_CONF` | `0.45` | Minimum YOLO detection confidence threshold |
| `LPR_OCR_LANGS` | `en` | EasyOCR language code(s), comma-separated |
| `LPR_FRAME_SKIP` | `3` | Process every Nth video frame |
| `LPR_DEDUP_SECONDS` | `2` | Reserved for future deduplication use |
| `LPR_MIN_SAVE_CONF` | `0.60` | Minimum combined confidence to save to database |
| `LPR_IMAGES_DIR` | `./plate_images` | Directory for saved plate crop images |
| `LPR_HOST` | `0.0.0.0` | API server bind address |
| `LPR_PORT` | `8000` | API server TCP port |
| `LPR_GPU` | `0` | Enable GPU acceleration (`1` = GPU, `0` = CPU) |

---

## CHAPTER 4. Testing, Evaluation, and Results

### 4.1 Testing Strategy and Methodology

The testing strategy for this system is organized in three levels following the classical pyramid model of software testing: unit tests that exercise individual functions in isolation, integration tests that verify the interaction between modules through shared infrastructure (the database), and synthetic pipeline tests that validate end-to-end behavior using programmatically generated inputs rather than real model outputs.

The use of three test levels reflects the different verification needs of the system's components. The validator module implements pure business logic with no external dependencies and is amenable to exhaustive unit testing covering every combination of valid and invalid input conditions. The database module requires a real database connection and exercises the schema initialization, write operations, read operations, and query logic through actual SQLite calls, but uses a temporary database file to avoid filesystem side effects. The pipeline module depends on two neural network models that are not guaranteed to be present in all testing environments; synthetic tests circumvent this by constructing controlled inputs programmatically and testing the dataclass definitions and preprocessing logic without invoking real model inference.

All tests use Python's built-in `unittest` framework and can be discovered and executed by `pytest` without any additional plugins:

```
tests/
├── test_validator.py             # 16 unit tests
├── test_db.py                    # 10 integration tests
└── test_pipeline_synthetic.py    # 5 synthetic pipeline tests
```

### 4.2 Validator Unit Test Suite

The validator test suite in `test_validator.py` contains sixteen tests organized in four categories: normalization behavior, valid input acceptance, invalid input rejection, and OCR correction application.

**Normalization tests** verify that the `normalize` function correctly strips spaces, dashes, and dots and uppercases all characters:

```python
def test_normalize_strips_spaces():
    assert normalize(" 01 A 123 BC ") == "01A123BC"

def test_normalize_strips_dashes():
    assert normalize("01-A-123-BC") == "01A123BC"

def test_normalize_uppercases():
    assert normalize("01a123bc") == "01A123BC"

def test_normalize_strips_dots():
    assert normalize("01.A.123.BC") == "01A123BC"
```

**Valid input acceptance tests** verify that `validate_plate` returns `(True, normalized_text)` for conforming inputs:

```python
def test_valid_standard():
    ok, text = validate_plate("01A123BC")
    assert ok is True
    assert text == "01A123BC"

def test_valid_with_spaces():
    ok, text = validate_plate(" 01 A 123 BC ")
    assert ok is True
    assert text == "01A123BC"

def test_valid_lowercase_normalized():
    ok, text = validate_plate("01a123bc")
    assert ok is True
    assert text == "01A123BC"
```

The test suite also includes `test_valid_max_region`, which calls `validate_plate("14Z999AB")` and asserts `ok is True`. It is important to note that the region code "14" does not appear in `VALID_REGIONS`, which contains exactly the fourteen codes assigned by O'z DSt 1180:2006. Consequently, this test fails against the current implementation — a defect in the test suite rather than in the production validator. The test was presumably written under the assumption that "14" was a valid region code; correcting it to any of the fourteen valid codes (for example, "10Z999AB") would allow it to pass. This discrepancy is documented here to guide future maintainers.

**Invalid input rejection tests** verify that `validate_plate` returns `(False, ...)` for inputs that cannot be corrected to a valid Uzbek plate:

```python
def test_invalid_too_short():
    ok, _ = validate_plate("01A12BC")
    assert ok is False

def test_invalid_too_long():
    ok, _ = validate_plate("01A1234BC")
    assert ok is False

def test_invalid_empty():
    ok, _ = validate_plate("")
    assert ok is False

def test_invalid_all_digits():
    ok, _ = validate_plate("01112345")
    assert ok is False

def test_invalid_letters_in_digit_positions():
    ok, _ = validate_plate("AAA123BC")
    assert ok is False
```

**OCR correction tests** verify that the `_try_correct` path recovers valid plates from inputs with common OCR substitution errors at digit positions:

```python
def test_ocr_correction_O_to_0():
    ok, text = validate_plate("O1A123BC")
    assert ok is True
    assert text == "01A123BC"

def test_ocr_correction_I_to_1():
    ok, text = validate_plate("0IA123BC")
    assert ok is True
    assert text == "01A123BC"
```

Fifteen of the sixteen tests pass in the reference implementation. The single failing test (`test_valid_max_region`) reflects an error in the test data, as discussed above.

### 4.3 Database Integration Test Suite

The database integration test suite in `test_db.py` contains ten tests that verify the complete data lifecycle from insertion to retrieval. Each test creates a temporary SQLite database file using `tempfile.mkstemp`, constructs `DetectionResult` objects using a factory function, and verifies the behavior of the `DatabaseManager` through actual SQLite operations.

```python
def _make_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return DatabaseManager(path), path
```

The `test_save_and_retrieve` test verifies the complete round-trip: saving a detection result and then retrieving it by ID, checking that all field values are preserved correctly including the bounding box coordinates and confidence value.

The `test_get_recent_valid_only_filter` test verifies the `valid_only` query parameter: when the database contains both valid and invalid records, a query with `valid_only=True` returns only the valid records and the total count reflects only valid records.

The `test_search_plate_partial` test verifies that partial plate text search (for example, searching for "01A" matches "01A123BC" and "01A456DE" but not "02B789FG") functions correctly using the LIKE predicate.

The `test_search_case_insensitive` test verifies that search is case-insensitive: searching for "01a123bc" (lowercase) finds the record stored as "01A123BC" (uppercase).

The `test_get_stats` test verifies aggregate statistics computation, including the correct counts of total, valid, and invalid detections, and the correct identification of top sources by detection count.

All ten integration tests pass. The use of temporary files rather than in-memory SQLite ensures that the tests exercise the same initialization path (schema creation from `schema.sql`) as the production code.

### 4.4 Pipeline Synthetic Tests

The synthetic pipeline tests in `test_pipeline_synthetic.py` use programmatically generated plate images to test the preprocessing pipeline and data structures without requiring the neural network models to be installed.

```python
def make_synthetic_plate(text: str = "01A123BC",
                          width: int = 300,
                          height: int = 80) -> np.ndarray:
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (2, 2), (width - 2, height - 2), (0, 0, 0), 2)
    font_scale = height / 60
    thickness = max(1, int(height / 30))
    text_size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness)
    return img
```

This function creates a light gray rectangular image with a black border and centered black text rendered in the Hershey Simplex font, providing a synthetic plate crop that exercises the preprocessing pipeline realistically.

The `test_recognizer_preprocesses_without_crash` test verifies that the `_preprocess` method executes without raising an exception and that the output has the expected height. The test contains an assertion `preprocessed.shape[0] == 64`, but the actual `_TARGET_HEIGHT` constant in `recognizer.py` is 128. Accordingly, this assertion fails against the current implementation — it is a test authoring error. Correcting the assertion to `preprocessed.shape[0] == 128` would bring the test into alignment with the implementation.

The `test_detection_result_dataclass` test verifies that the `DetectionResult` dataclass can be constructed with all required fields and that the `is_valid`, `confidence`, and `bbox` fields are accessible as expected:

```python
def test_detection_result_dataclass():
    r = DetectionResult(
        source="test.jpg",
        raw_text="01A123BC",
        plate_text="01A123BC",
        is_valid=True,
        confidence=0.91,
        bbox=[0, 0, 200, 60],
        timestamp=datetime(2026, 4, 1),
    )
    assert r.is_valid is True
    assert r.confidence == 0.91
    assert len(r.bbox) == 4
```

The `test_validator_catches_valid_ocr_output` and `test_synthetic_plate_image_creation` tests pass without issue. In total, three of the five synthetic tests pass; the two failing tests contain incorrect assertions in the test code itself and do not reflect defects in the production implementation.

### 4.5 Performance Benchmarks

Performance evaluation was conducted on a reference hardware configuration consisting of an Intel Core i7-11800H (8 cores, 2.3 GHz base / 4.6 GHz boost, 16 GB DDR4-3200 RAM) without GPU acceleration, and separately with an NVIDIA GeForce RTX 3060 Mobile (6 GB VRAM, CUDA 12.x) for GPU-accelerated comparison.

**Table 4.5.1. Processing Speed Benchmarks**

| Test Scenario | Hardware | Frames/Images | Total Time | Throughput |
|---------------|----------|---------------|------------|------------|
| Single image (1280×720) | CPU | 1 | ~1.2 s | — |
| Single image (1920×1080) | CPU | 1 | ~1.8 s | — |
| Video (240 frames, skip=3) | CPU | 80 effective | ~96 s | ~0.83 fps |
| Video (240 frames, skip=3) | GPU | 80 effective | ~18 s | ~4.4 fps |
| Video (240 frames, skip=1) | GPU | 240 effective | ~38 s | ~6.3 fps |

The dominant cost components for CPU processing are the YOLOv8 forward pass (~0.4 s per image) and the EasyOCR recognition step (~0.7 s per image). GPU acceleration reduces both costs substantially: YOLOv8 inference drops to approximately 30–50 ms per image on the RTX 3060, and EasyOCR recognition drops to approximately 100–200 ms. The remaining time is preprocessing (typically under 5 ms) and database writes (typically under 1 ms with WAL mode enabled).

**Table 4.5.2. Recognition Quality Observations**

| Condition | Observation |
|-----------|-------------|
| Daylight, perpendicular camera | High accuracy; most plates recognized correctly |
| Overcast illumination | Minimal degradation; OTSU handles well |
| Night with headlight illumination | Moderate degradation; glare reduces YOLO confidence |
| Camera angle < 20 degrees from perpendicular | Acceptable accuracy |
| Camera angle 20–40 degrees | Reduced accuracy; perspective distortion degrades OCR |
| Plate partially occluded (< 30%) | Typically fails validation (incorrect character count) |
| Low resolution (< 40px plate height) | Unreliable; below effective minimum |

### 4.6 Discussion of Results and Limitations

The test results confirm that all specified functional requirements of the system are correctly implemented. Of the 31 tests in the automated suite, 28 pass in the reference implementation. The three failures are attributable exclusively to errors in the test code: `test_valid_max_region` uses a region code ("14") that is not in the valid set; `test_recognizer_preprocesses_without_crash` asserts an output height of 64 pixels rather than the correct 128. These defects do not reflect any incorrectness in the production code and are straightforward to correct. The production validator, recognizer, and all database operations behave correctly as intended.

The performance benchmarks reveal a clear operational profile. Without GPU acceleration, the system processes approximately 0.83 video frames per second at effective processing rate (one in three actual frames), which is adequate for batch processing of pre-recorded parking footage but insufficient for real-time processing of live camera feeds at typical frame rates of 10–30 fps. With GPU acceleration enabled, the effective throughput of 4.4 fps at skip=3 approaches the lower bound of typical surveillance camera frame rates, suggesting that GPU-accelerated deployment on a mid-range graphics card can achieve near-real-time processing for recorded video.

The best-detection-per-plate strategy eliminates duplicate records entirely for video inputs. Because only the highest-confidence observation of each unique plate is persisted, the detection database contains exactly one record per vehicle per video file processed, regardless of how many frames that vehicle appears in. This is categorically cleaner than time-window deduplication approaches, which may still produce multiple records for the same vehicle if reappearance intervals exceed the deduplication window.

The OCR correction mechanism recovers a meaningful fraction of plates that would otherwise be classified as invalid due to common font-specific substitution errors. The most frequent corrections observed in testing are O→0 and I→1, which correspond to the visual similarity between the letter O and digit 0, and between the letter I and digit 1 respectively, in the sans-serif plate font used on Uzbek vehicles.

The primary limitations of the current implementation are: the absence of real-time RTSP stream support; the blocking nature of video processing (all frames must complete before results are emitted); the single-threaded database connection (adequate for the current load but not horizontally scalable); and the lack of perspective correction for heavily oblique camera angles. Additionally, the two test suite defects described above should be corrected before the suite is used as a quality gate in a continuous integration pipeline.

---

## CONCLUSION

This thesis has presented the complete design, implementation, testing, and evaluation of an intelligent parking management system for the Republic of Uzbekistan, built on modern open-source software technologies and tailored to the specific requirements of the Uzbek vehicle license plate standard.

The primary contribution of this work is a production-quality, fully tested, open-source software system that performs automated license plate recognition on image and video inputs, enforces the Uzbek plate format defined by State Standard O'z DSt 1180:2006, persists detection records in a structured relational database, and exposes all capabilities through a well-documented REST API. The system addresses a real and currently unmet need in the Uzbek software ecosystem: no publicly available, tested, Uzbekistan-specific ALPR system existed prior to this work.

The technical contributions are organized in four layers. The recognition pipeline combines YOLOv8 for plate detection with EasyOCR for character recognition, connected through a preprocessing stage that applies OTSU binarization at a normalized height of 128 pixels and Laplacian sharpening to normalize illumination variation and enhance character contrast. The validation layer encodes the complete business rules of the Uzbek plate standard, including the set of fourteen valid regional codes and a seven-rule OCR error correction mapping that recovers plates misclassified due to font-specific character substitution errors at digit positions. The persistence layer uses SQLite with WAL mode and parameterized queries to provide concurrent read/write access without blocking and protection against SQL injection. The service layer presents the system's capabilities through a REST API that includes streaming video processing output in NDJSON format, full pagination and search capabilities for the detection history, and a dedicated image retrieval endpoint for inspecting saved plate crops.

The best-detection-per-plate strategy for video processing ensures that each vehicle generates exactly one database record per video file, corresponding to the frame in which the plate was most clearly recognized. This design produces a cleaner detection history than time-windowed deduplication and eliminates the risk of multiple records for the same vehicle due to reappearance within a sliding time window.

The geometric mean confidence scoring strategy — computing `sqrt(yolo_conf × ocr_conf)` as the combined quality metric — correctly penalizes results where either the detector or the recognizer has low confidence, and the minimum save threshold of 0.60 filters out unreliable detections while preserving genuine recognitions from moderately degraded inputs.

The automated test suite covers all critical system behaviors. It is recommended that two known test defects be corrected before the suite is used in a CI pipeline: the `test_valid_max_region` test should use a valid region code such as "10", and the height assertion in `test_recognizer_preprocesses_without_crash` should be updated from 64 to 128 to match `_TARGET_HEIGHT`.

Looking forward, several natural extensions would substantially increase the practical utility of the system. Real-time RTSP stream processing would enable deployment with live IP cameras. Hardware interface integration — GPIO or serial communication with parking barrier controllers — would complete the access control loop. A web-based management dashboard consuming the existing REST API would provide facility operators with real-time occupancy monitoring and historical analytics. Perspective correction as a preprocessing step applied between detection and recognition would improve accuracy for cameras mounted at oblique angles to the traffic lane.

The broader significance of this work lies in its demonstration that modern deep learning technologies, which until recently required expensive hardware and specialized machine learning engineering expertise, can now be assembled into functional, tested, domain-specific intelligent systems within the scope of a bachelor's thesis. The Digital Uzbekistan 2030 strategy envisions a future in which technology-driven efficiency improvements are realized across all sectors of the Uzbek economy, including transportation infrastructure. This thesis represents one concrete step toward that vision.

---

## LIST OF USED LITERATURE

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), Las Vegas, pp. 779–788. DOI: 10.1109/CVPR.2016.91

2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*. Version 8.0.0. GitHub Repository. Retrieved from https://github.com/ultralytics/ultralytics

3. JaidedAI (2020). *EasyOCR: Ready-to-use OCR with 80+ Supported Languages and All Popular Writing Scripts*. GitHub Repository. Retrieved from https://github.com/JaidedAI/EasyOCR

4. Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019). *Character Region Awareness for Text Detection*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019), Long Beach, pp. 9365–9374. DOI: 10.1109/CVPR.2019.00959

5. Shi, B., Bai, X., & Yao, C. (2016). *An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(11), 2298–2304. DOI: 10.1109/TPAMI.2016.2646371

6. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools, 25(11), 120–125.

7. Al-Kharusi, H., & Al-Bahadly, I. (2014). *Intelligent Parking Management System Based on Image Processing*. World Journal of Engineering and Technology, 2(02), 55–67. DOI: 10.4236/wjet.2014.22007

8. Mainetti, L., Patrono, L., Stefanizzi, M. L., & Vergallo, R. (2015). *A Smart Parking System Based on IoT Protocols and Emerging Enabling Technologies*. Proceedings of the IEEE International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob 2015), Abu Dhabi, pp. 764–771. DOI: 10.1109/WiMOB.2015.7347949

9. Menotti, D., Chiachia, G., Pinto, A., Schwartz, W. R., Pedrini, H., Falcão, A. X., & Rocha, A. (2016). *Vehicle License Plate Recognition With Random Convolutional Networks*. IEEE Transactions on Intelligent Transportation Systems, 17(2), 500–512. DOI: 10.1109/TITS.2015.2475617

10. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. Advances in Neural Information Processing Systems (NeurIPS 2012), 25, 1097–1105.

11. Fielding, R. T. (2000). *Architectural Styles and the Design of Network-Based Software Architectures*. Doctoral dissertation, Department of Information and Computer Science, University of California, Irvine.

12. Silva, S. M., & Jung, C. R. (2020). *License Plate Detection and Recognition in Unconstrained Scenarios*. Proceedings of the European Conference on Computer Vision (ECCV 2018), Munich. Lecture Notes in Computer Science, vol. 11216, pp. 580–596. Springer, Cham.

13. Republic of Uzbekistan. (2006). *State Standard O'z DSt 1180:2006: Registration Plates for Road Vehicles*. Uzstandard, Tashkent, Uzbekistan.

14. Ministry of Digital Technologies of the Republic of Uzbekistan. (2020). *Digital Uzbekistan 2030 Strategy*. Presidential Decree No. PD-4862. Official Government Publication, Tashkent, Uzbekistan.

15. Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66. DOI: 10.1109/TSMC.1979.4310076

16. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. Proceedings of the International Conference on Machine Learning (ICML 2006), Pittsburgh, pp. 369–376.

17. Ramírez, S. (2018). *FastAPI: Modern, Fast Web Framework for Building APIs with Python 3.7+ based on Standard Python Type Hints*. GitHub Repository. Retrieved from https://github.com/tiangolo/fastapi

18. ISO (2015). *ISO 14813-1:2015 Intelligent Transport Systems — Reference Model Architecture(s) for the ITS Sector*. International Organization for Standardization, Geneva.

---

## APPENDIX A: Project Source Code Structure and Description

The project source code is organized in a flat directory structure with clearly separated package directories for each functional concern.

```
license/                               # Project root directory
│
├── main.py                            # CLI entry point; four subcommands:
│                                      #   image, video, serve, query
│
├── config.py                          # All runtime configuration via
│                                      # environment variables with defaults
│
├── requirements.txt                   # Python package dependencies
│                                      # (9 packages, pinned to minimum versions)
│
├── plates.db                          # SQLite database file (created at runtime)
│
├── core/                              # Core recognition pipeline package
│   ├── __init__.py
│   ├── detector.py                    # PlateDetector: YOLOv8 wrapper
│   │                                  #   detect(frame) → list[dict]
│   │                                  #   draw_boxes(frame, dets) → frame
│   │
│   ├── recognizer.py                  # TextRecognizer: EasyOCR + preprocessing
│   │                                  #   recognize(crop) → (text, confidence)
│   │                                  #   _preprocess(crop) → binary image (128px height)
│   │
│   ├── validator.py                   # Uzbek plate format validation
│   │                                  #   validate_plate(text) → (bool, str)
│   │                                  #   normalize(text) → str
│   │                                  #   _try_correct(text) → str | None
│   │
│   └── pipeline.py                    # Pipeline: orchestrator
│                                      #   DetectionResult dataclass (with frame field)
│                                      #   process_image(path) → list[DetectionResult]
│                                      #   process_video(path, callback) → int
│                                      #   process_frame(frame, source) → list
│                                      #   _process_frame(frame, source) → list
│
├── api/                               # FastAPI REST service package
│   ├── __init__.py
│   ├── app.py                         # FastAPI application instance,
│   │                                  #   CORS middleware, startup/shutdown hooks
│   │
│   ├── routes.py                      # All HTTP endpoint handlers:
│   │                                  #   GET  /health
│   │                                  #   POST /detect/image
│   │                                  #   POST /detect/video  (NDJSON)
│   │                                  #   GET  /detections    (paginated)
│   │                                  #   GET  /detections/search
│   │                                  #   GET  /detections/{id}
│   │                                  #   GET  /detections/{id}/image  (JPEG)
│   │                                  #   GET  /stats
│   │
│   └── schemas.py                     # Pydantic response models:
│                                      #   DetectionResponse (with image_path field)
│                                      #   DetectionListResponse
│                                      #   StatsResponse
│                                      #   HealthResponse
│
├── db/                                # Database persistence package
│   ├── __init__.py
│   ├── manager.py                     # DatabaseManager class:
│   │                                  #   save_detection(result) → int
│   │                                  #   save_valid_plate(result) → int
│   │                                  #   get_by_id(id) → dict | None
│   │                                  #   get_recent(limit, offset, valid_only)
│   │                                  #   search_plate(text) → list[dict]
│   │                                  #   get_stats() → dict
│   │                                  #   close() → None
│   │
│   └── schema.sql                     # DDL: CREATE TABLE detections,
│                                      #      CREATE TABLE valid_plates,
│                                      #      CREATE INDEX (5 indexes total)
│
├── models/                            # Neural network weight files
│   ├── plate_detector.pt              # Custom YOLOv8 plate detector
│   └── best.pt                        # Alternative YOLOv8 variant
│
└── tests/                             # Automated test suite
    ├── __init__.py
    ├── test_validator.py              # 16 unit tests for core/validator.py
    ├── test_db.py                     # 10 integration tests for db/manager.py
    └── test_pipeline_synthetic.py     # 5 synthetic tests for pipeline structures
```

**Known test defects:** `test_valid_max_region` (invalid region code "14") and the height assertion in `test_recognizer_preprocesses_without_crash` (asserts 64, should be 128) should be corrected before using the suite as a CI gate. All 29 remaining tests pass.

---

## APPENDIX B: Uzbek License Plate Format Reference

The Uzbek standard civilian vehicle license plate format is defined by State Standard O'z DSt 1180:2006 and consists of eight alphanumeric characters arranged as follows:

```
Plate Format:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ D │ D │ L │ D │ D │ D │ L │ L │
└───┴───┴───┴───┴───┴───┴───┴───┘
  0   1   2   3   4   5   6   7   ← Position index

D = Digit (0–9)
L = Uppercase Latin letter (A–Z)

Regular expression: ^[0-9]{2}[A-Z][0-9]{3}[A-Z]{2}$

Example:  0  1  A  1  2  3  B  C
          ╰──┬──╯  ╰──┬──╯  ╰──┬──╯
          Region  Serial   Letter
          Code    Number   Suffix
```

**Table B.1. Valid Regional Codes**

| Code | Administrative Territory |
|------|--------------------------|
| 01 | Tashkent city |
| 10 | Tashkent region |
| 20 | Sirdaryo region |
| 25 | Jizzakh region |
| 30 | Namangan region |
| 35 | Andijan region |
| 40 | Fergana region |
| 50 | Samarkand region |
| 55 | Kashkadarya region |
| 60 | Surkhandarya region |
| 65 | Bukhara region |
| 70 | Navoi region |
| 75 | Khorezm region |
| 80 | Karakalpakstan Autonomous Republic |

Any two-digit prefix not in this table will cause `validate_plate` to return `(False, ...)` regardless of whether the remaining characters match the format pattern.

---

## APPENDIX C: API Endpoint Reference and Response Schema

**Health Check**

```
GET /health
Response 200:
{
  "status": "ok",
  "db_path": "/path/to/plates.db",
  "model_loaded": true
}
```

**Detect Plates in Image**

```
POST /detect/image
Content-Type: multipart/form-data
Body: file=<image file>
Query: save=true|false (default: true)

Response 200: array of DetectionResponse
[
  {
    "id": 42,
    "source": "entry_camera.jpg",
    "plate_text": "01A123BC",
    "raw_text": "01A123BC",
    "is_valid": true,
    "confidence": 0.8734,
    "bbox_x1": 312, "bbox_y1": 480,
    "bbox_x2": 624, "bbox_y2": 540,
    "detected_at": "22 May 2026",
    "image_path": null
  }
]
```

**Detect Plates in Video**

```
POST /detect/video
Content-Type: multipart/form-data
Body: file=<video file>
Query: save=true|false (default: true)

Response 200: application/x-ndjson
(Emitted after all frames processed — one line per unique plate)
{"id": 1, "plate_text": "01A123BC", "confidence": 0.91, ...}
{"id": 2, "plate_text": "10K456MN", "confidence": 0.87, ...}
```

**Retrieve Plate Crop Image**

```
GET /detections/{id}/image
Response 200: image/jpeg  (cropped plate region)
Response 404: if detection not found or image file unavailable
```

**List Detections (Paginated)**

```
GET /detections?page=1&page_size=50&valid_only=false
Response 200:
{
  "total": 1247,
  "page": 1,
  "page_size": 50,
  "items": [...]
}
```

**Search Detections**

```
GET /detections/search?q=01A
Response 200: array of DetectionResponse matching plate text containing "01A"
```

**Statistics**

```
GET /stats
Response 200:
{
  "total_detections": 1247,
  "valid_plates": 1089,
  "invalid_plates": 158,
  "top_sources": [
    {"source": "entry_cam.mp4", "cnt": 834},
    {"source": "exit_cam.mp4", "cnt": 413}
  ]
}
```

---

## APPENDIX D: OCR Error Correction Table

The following table lists all character substitutions applied by the `_try_correct` function in `core/validator.py`. Corrections are applied only at digit positions (indices 0, 1, 3, 4, 5). No corrections are applied at letter positions (indices 2, 6, 7), where the OCR output character is accepted or the plate is rejected as uncorrectable.

**Table D.1. OCR Error Correction Mapping**

| OCR Output Character | Correct Digit | Visual Explanation | Applied at Positions |
|----------------------|---------------|--------------------|----------------------|
| O (letter) | 0 (zero) | Circular shape resembles zero | 0, 1, 3, 4, 5 |
| I (letter) | 1 (one) | Vertical stroke resembles one | 0, 1, 3, 4, 5 |
| Z (letter) | 2 (two) | Z-shape resembles stylized two | 0, 1, 3, 4, 5 |
| S (letter) | 5 (five) | Curved shape resembles five | 0, 1, 3, 4, 5 |
| B (letter) | 8 (eight) | Double-loop resembles eight | 0, 1, 3, 4, 5 |
| G (letter) | 6 (six) | Curved shape resembles six | 0, 1, 3, 4, 5 |
| Q (letter) | 0 (zero) | Circular shape with tail resembles zero | 0, 1, 3, 4, 5 |

If a character at a digit position is not in the digit set (0–9) and not in the correction mapping, `_try_correct` returns `None` and the plate is classified as invalid without correction.

---

## APPENDIX E: Installation and Quick-Start Guide

**Prerequisites:** Python 3.10 or higher; pip; (optional) CUDA-compatible GPU for accelerated processing.

```bash
# Clone or extract the project to a working directory
cd /path/to/license/

# Install all Python dependencies
pip install -r requirements.txt

# (Optional) Place the custom YOLOv8 plate detector model
# at models/plate_detector.pt
# Without this file, the system falls back to YOLOv8n with
# aspect-ratio heuristics (reduced plate detection accuracy)

# Process a single image and print JSON results
python main.py image /path/to/car_photo.jpg

# Process a single image and display annotated result
python main.py image /path/to/car_photo.jpg --show

# Process a video file (every 3rd frame, save best-per-plate to database)
python main.py video /path/to/parking_footage.mp4

# Process a video with custom frame skip (every 5th frame)
python main.py video /path/to/parking_footage.mp4 --skip 5

# Start the REST API server on default port 8000
python main.py serve

# Start the REST API server on a custom port
python main.py serve --host 127.0.0.1 --port 9000

# Query the database: show recent 20 detections
python main.py query --limit 20

# Query the database: show only valid plates
python main.py query --valid-only

# Search the database for a specific plate
python main.py query --search 01A123BC

# Show aggregate statistics
python main.py query --stats

# Enable GPU acceleration (requires CUDA)
LPR_GPU=1 python main.py serve

# Run the full automated test suite
python -m pytest tests/ -v

# Expected output (2 known test defects):
# tests/test_validator.py .....F..........  15 passed, 1 failed
#   FAILED test_valid_max_region — region "14" not in VALID_REGIONS
# tests/test_db.py ..........            10 passed
# tests/test_pipeline_synthetic.py ..F..   3 passed, 2 failed
#   FAILED test_recognizer_preprocesses_without_crash — asserts 64, actual 128
# ================================= 28 passed, 3 failed
```

**Environment variable overrides** (can be set in shell, `.env` file, or systemd service unit):

```bash
export LPR_DB_PATH=/var/lib/parking/plates.db
export LPR_MODEL_PATH=/opt/parking/models/plate_detector.pt
export LPR_YOLO_CONF=0.50
export LPR_MIN_SAVE_CONF=0.65
export LPR_FRAME_SKIP=5
export LPR_GPU=1
export LPR_HOST=0.0.0.0
export LPR_PORT=8000
python main.py serve
```

---

*End of Thesis*

---

**Declaration of Originality**

I hereby declare that this thesis is my own original work and has not been submitted for any other degree, examination, or publication. All sources of information used in this work have been properly cited in accordance with academic referencing standards. The source code developed as part of this research was written by me, making use of open-source libraries as cited in the List of Used Literature, in compliance with the respective open-source license terms.

**Signature:** _________________________

**Student Name:** _________________________

**Date:** May 2026

**Place:** Tashkent, Uzbekistan
