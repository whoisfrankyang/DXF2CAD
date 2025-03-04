import os
import adsk.core, adsk.fusion, traceback

def run(context):
    ui = None

    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        documents = app.documents
        documents_to_close = []

        for i in range(documents.count):
            doc = documents.item(i)
            if doc.name != "Untitled":
                documents_to_close.append(doc)

        for doc in documents_to_close:
            doc.close(False)
        # ui.messageBox(f"Closed {len(documents_to_close)} open document(s) that were not named 'Untitled'.")

        # Make sure to customize the following paths
        # must use absolute path   
        stp_folder = '/Users/frankyang/Desktop/Pixelate/Blueprint2CAD/Fusion_Reconstruction/filter_step_filess/folder_009'
        output_folder = '/Users/frankyang/Desktop/Pixelate/Blueprint2CAD/design_generation/pdf_data/009-pdf'

        if not os.path.exists(stp_folder):
            ui.messageBox(f"STEP folder does not exist: {stp_folder}")
            return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        stp_files = [f for f in os.listdir(stp_folder) if f.lower().endswith('.step') or f.lower().endswith('.stp')]

        if not stp_files:
            ui.messageBox(f"No STEP files found in folder: {stp_folder}")
            return
        # else:
            # ui.messageBox(f"Found {len(stp_files)} STEP files in folder: {stp_folder}")

        existing_pdfs = [f for f in os.listdir(output_folder) if f.lower().endswith('.pdf')]
        processed_files = [f.split(' Drawing v1.pdf')[0] for f in existing_pdfs]


        unprocessed_files = []
        for stp_file in stp_files:
            base_name = os.path.splitext(stp_file)[0]
            if base_name not in processed_files:
                unprocessed_files.append(stp_file)

        if not unprocessed_files:
            ui.messageBox("All STEP files have been processed.")
            return

        file_name = unprocessed_files[0]
        stp_file_path = os.path.join(stp_folder, file_name)
        ui.messageBox(f"Processing file: {file_name}")

        import_manager = app.importManager
        stp_options = import_manager.createSTEPImportOptions(stp_file_path)
        success = import_manager.importToNewDocument(stp_options)

        if not success:
            ui.messageBox(f"Failed to import STEP file: {stp_file_path}")
            return
        # else:
        #     ui.messageBox(f"Imported STEP file {file_name} successfully.")

        doc = app.activeDocument
        design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))

        if design:
            # ui.messageBox("Design accessed successfully.")

            doc.name = os.path.splitext(file_name)[0]

            cmd_def = ui.commandDefinitions.itemById('NewFusionDrawingDocumentCommand')
            if not cmd_def:
                ui.messageBox("Failed to get command definition for creating drawing.")
                doc.close(False)
                ui.messageBox("Closed design document.")
                return

            cmd_def.execute()
            adsk.doEvents()
            # ui.messageBox("Drawing creation command executed.")

            # ui.messageBox("Please complete the drawing creation manually. After you are done, export the drawing as PDF to the output folder.")

            # Exit the script
            return

        else:
            ui.messageBox("Failed to access design.")
            # Close the imported design document
            doc.close(False)
            ui.messageBox("Closed design document.")

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
