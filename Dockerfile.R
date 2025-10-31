# Use official R base image
FROM r-base:4.2.2

# Set working directory inside container
WORKDIR /app

# Copy install script first and install packages
COPY src/app-r/install_packages.R ./install_packages.R
RUN Rscript install_packages.R

# Copy the actual R program
COPY src/app-r/app.R ./app.R

# Default command to run the script
CMD ["Rscript", "app.R"]
