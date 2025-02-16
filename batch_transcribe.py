import os
import whisper
import subprocess
import tempfile
from datetime import datetime
from openai import OpenAI

# Configuración de OpenAI
client = OpenAI(api_key="api_key_here")


txtPromptOpenAI = "Analiza al 100%% la transcripción del video titulado |archivo| donde se habla de auditoría y calidad como si fueses Alex Hormozi. Dame un análisis detallado y super completo del video basado en la siguente Transcripción Completa: |texto|"
txtRoleOpenAI = """Eres Alex Hormozi y quiero que transcribas los videos de Youtube sobre auditoría y calidad detalladamente y completamente sin dejarte ningún detalle importante en base a la transcripción proporcionada."""

txtOutputInicioConf = "\n=== Configuración Inicial ==="
txtRutaCarpeta = "👉 Ruta completa de la carpeta con archivos: "
txtMsgCarpetaInvalida = "❌ ¡La carpeta no existe! Intenta nuevamente"
txtIdiomaTranscripcion= "👉 Idioma para transcripción (ej: es, en, fr): "
txtMsgErrorIdiomaTranscripcion = "❌ ¡Idioma inválido! Debe ser un código de 2 letras"
txtOutputResumenAnalizados = "\n=== Resumen de Archivos Analizados ===\n"
txtOutputProcesoTransAnali = "\n=== Proceso de Transcripción y Análisis ==="
txtOutputResFinal = "\n=== Resumen Final ==="

nombreModeloWhisper = 'base' # Modelo de transcripción (medium.pt, base.pt, large-v3-turbo.pt)

def obtener_datos_usuario():
    print(txtOutputInicioConf)
    
    while True:
        carpeta = input(txtRutaCarpeta).strip()
        if os.path.isdir(carpeta):
            break
        print(txtMsgCarpetaInvalida)
    
    while True:
        idioma = input(txtIdiomaTranscripcion).strip().lower()
        if len(idioma) == 2 and idioma.isalpha():
            break
        print(txtMsgErrorIdiomaTranscripcion)

    return carpeta, idioma

def inicializar_txt(carpeta):
    txt_path = os.path.join(carpeta, f"resumenes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txtOutputResumenAnalizados)
    return txt_path

def procesar_archivos(carpeta, idioma, txt_path):
    modelo = whisper.load_model(nombreModeloWhisper)
    total = sum(1 for f in os.listdir(carpeta) if es_archivo_valido(f))
    procesados = 0
    errores = 0
    
    print(txtOutputProcesoTransAnali)
    print(f"📂 Carpeta seleccionada: {carpeta}")
    print(f"🌐 Idioma seleccionado: {idioma}")
    print(f"💾 Archivo TXT: {txt_path}\n")

    for archivo in os.listdir(carpeta):
        if not es_archivo_valido(archivo):
            continue

        ruta_completa = os.path.join(carpeta, archivo)

        nombreArchivo = os.path.splitext(archivo)[0]
        ruta_transcripcion = os.path.join(carpeta, nombreArchivo + ".txt")
        

        temp_audio_path = None
        
        try:
            print(f"🔍 Procesando {procesados+1}/{total}: {archivo}")

            if os.path.isfile(ruta_transcripcion) :
                print(f"❕ Transcripción ya existe para {archivo}")
                procesados += 1

                with open(ruta_transcripcion, 'r', encoding='utf-8') as file:
                    texto = file.read()

            else:

                # Crear archivo temporal
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name

                # Convertir archivo
                subprocess.run([
                    'ffmpeg', '-i', ruta_completa, '-vn',
                    '-acodec', 'pcm_s16le', '-ar', '16000',
                    '-ac', '1', '-threads', '0',
                    temp_audio_path, '-y', '-loglevel', 'error'
                ], check=True)
                    
                # Transcribir
                resultado = modelo.transcribe(
                    temp_audio_path,
                    language=idioma,
                    fp16=False,
                    verbose=False
                )
                
                texto = resultado['text'].strip()

            # Escribir transcripción en TXT
            with open(ruta_transcripcion, 'a', encoding='utf-8') as f:
                f.write(f"📁 Archivo: {archivo}\n")
                f.write(f"📝 Transcripción: {texto}\n")
                f.write("-"*50 + "\n\n")
            
            # Analizar con OpenAI
            prompt = txtPromptOpenAI.replace("|archivo|", archivo).replace("|texto|", texto)

            '''
            prompt = f"""
            Analiza al 100% la transcripción del video titulado "{archivo}" donde se habla de auditoría y calidad como si fueses Alex Hormozi. Dame un análisis detallado y super completo del video basado en la siguente Transcripción Completa:
            {texto}
            """
            '''

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": txtRoleOpenAI
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1800,
                top_p=0.9,
                frequency_penalty=0.4,
                presence_penalty=0.25
            )
            
            resumen = response.choices[0].message.content.strip()
            
            # Escribir en TXT
            with open(txt_path, 'a', encoding='utf-8') as f:
                f.write(f"📁 Archivo: {archivo}\n")
                f.write(f"📝 Resumen: {resumen}\n")
                f.write("-"*50 + "\n\n")
            
            print(f"✅ Análisis exitoso: {archivo}")
            procesados += 1

        except subprocess.CalledProcessError as e:
            errores += 1
            print(f"❌ Error de conversión de audio en {archivo}")
            
        except Exception as e:
            errores += 1
            print(f"❌ Error procesando {archivo}: {str(e)}")
            
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    return procesados, errores

def es_archivo_valido(archivo):
    extensiones = ('.mp3', '.wav', '.mp4', '.avi', 
                  '.mov', '.m4a', '.flac', '.mkv', '.aac')
    return archivo.lower().endswith(extensiones)

def mostrar_resumen(procesados, errores, txt_path):
    print(txtOutputResFinal)
    print(f"✅ Archivos procesados correctamente: {procesados}")
    print(f"❌ Archivos con errores: {errores}")
    print(f"💾 Archivo TXT generado: {txt_path}")
    print("\n⚠️ Revisa los resultados en el archivo TXT generado")

if __name__ == '__main__':
    try:
        carpeta, idioma = obtener_datos_usuario()
        txt_path = inicializar_txt(carpeta)
        procesados, errores = procesar_archivos(carpeta, idioma, txt_path)
        mostrar_resumen(procesados, errores, txt_path)
    except Exception as e:
        print(f"❌¡Error crítico! {str(e)}")