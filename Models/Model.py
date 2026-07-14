import segmentation_models_pytorch as smp

# Dictionary mapping decoder names to their corresponding classes in the segmentation_models_pytorch library.
DECODERS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "segformer": smp.Segformer
}

def build_segmentation_model(
    decoder_name="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    activation=None,
    classes=1
):
    """
    Cria qualquer modelo de segmentação do SMP e sua função de pré-processamento.
    """
    # Normalize the decoder name to lowercase for case-insensitive matching
    name_lower = decoder_name.lower()
    
    if name_lower not in DECODERS:
        disponiveis = ", ".join(DECODERS.keys())
        raise ValueError(f"Decoder '{decoder_name}' não suportado. Escolha entre: {disponiveis}")
    
    # Search for the corresponding model class based on the decoder name
    model_class = DECODERS[name_lower]
    
    # Instatiate the model with the specified parameters
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=activation
    )
    
    # Search the preprocessing function for the specified encoder and weights
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_name, 
        encoder_weights
    )
    
    print(model)

    return model, preprocessing_fn