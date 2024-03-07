FROM python:3.10
COPY src /app
RUN chmod +x /app/script.sh
RUN /app/script.sh
RUN pip install numpy
CMD python3 /app/run_gen.py