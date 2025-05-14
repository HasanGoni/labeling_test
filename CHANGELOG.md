# Changelog

All notable changes to this project will be documented in this file.
## [0.0.2] - 2025-05-14

### Added
- **LBM Models**: Integration of LBM models for relighting, depth, and normal maps.

## [0.0.1] - 2024-XX-XX

### Added
- **Personalized SAM (PerSAM) Integration**: One-shot segmentation using HuggingFace's SAM model, with functions for embedding, mask refinement, and similarity computation.
- **Grounding DINO Labeling**: Zero-shot object detection and segmentation pipeline using Grounding DINO and SAM, with polygon refinement and annotated image output.
- **SegGPT Support**: Prompt-based segmentation using SegGPT, with mask overlay and saving utilities.
- **High-Quality SAM Inference**: Visualization utilities for points and masks, leveraging HQ-SAM for improved mask quality.
- **Image Tiling**: Utility for processing large images in tiles.
- **Core Utilities**: Foundational helpers for the labeling workflow.

### Improved
- Modularized codebase for easy extension and experimentation with new segmentation models.
- Clear separation between data loading, model inference, and visualization.

### Documentation
- Example-rich README with installation, usage, and troubleshooting tips.
- Inline docstrings and type hints for all major functions.

### Project Metadata
- Author: Hasan
- License: Apache 2.0
- Python ≥ 3.7
- Initial dependencies: `fastcore`, `pandas`, `transformers`, `opencv-python`, `torch`

---

**Note:**  
This is the initial release. All APIs and features are subject to change as the project evolves. 