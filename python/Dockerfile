# Dockerfile for tangent_py development and testing
# Uses pip wheels for cppyy (pre-built for linux/x86_64)

FROM python:3.10-slim

# g++ required by cppyy's Cling JIT at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

# Install cppyy and test dependencies via pip
RUN pip install --no-cache-dir cppyy numpy pytest

# Verify cppyy loads correctly
RUN python -c "import cppyy; print('cppyy', cppyy.__version__)"

# Set working directory
WORKDIR /tangent

# Copy the project
COPY . .

# Install tangent_py in editable mode
RUN pip install --no-cache-dir -e python/

# Default command: run tests
CMD ["pytest", "python/tests/", "-v"]
