import segmentation_models_pytorch as smp

# Criamos um dicionário mapeando nomes para as classes do SMP
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
    # 1. Normaliza o nome para evitar problemas com maiúsculas/minúsculas
    name_lower = decoder_name.lower()
    
    if name_lower not in DECODERS:
        disponiveis = ", ".join(DECODERS.keys())
        raise ValueError(f"Decoder '{decoder_name}' não suportado. Escolha entre: {disponiveis}")
    
    # 2. Busca a classe do modelo correspondente (ex: smp.Unet ou smp.FPN)
    model_class = DECODERS[name_lower]
    
    # 3. Instancia o modelo dinamicamente
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=activation
    )
    
    # 4. Busca a função de pré-processamento (que depende apenas do encoder)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_name, 
        encoder_weights
    )
    
    print(model)

    return model, preprocessing_fn

build_segmentation_model(decoder_name="unet", encoder_name="resnet34", encoder_weights="imagenet", activation=None, classes=1)