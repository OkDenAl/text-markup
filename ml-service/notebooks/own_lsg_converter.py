from transformers import AutoTokenizer
import json
import warnings
import torch
import sys

label2id = {
    'O': 0,
    'B-AGE': 1,
    'I-AGE': 2,
    'B-AWARD': 3,
    'I-AWARD': 4,
    'B-CITY': 5,
    'I-CITY': 6,
    'B-COUNTRY': 7,
    'I-COUNTRY': 8,
    'B-CRIME': 9,
    'I-CRIME': 10,
    'B-DATE': 11,
    'I-DATE': 12,
    'B-DISEASE': 13,
    'I-DISEASE': 14,
    'B-DISTRICT': 15,
    'I-DISTRICT': 16,
    'B-EVENT': 17,
    'I-EVENT': 18,
    'B-FACILITY': 19,
    'I-FACILITY': 20,
    'B-FAMILY': 21,
    'I-FAMILY': 22,
    'B-IDEOLOGY': 23,
    'I-IDEOLOGY': 24,
    'B-LANGUAGE': 25,
    'I-LANGUAGE': 26,
    'B-LAW': 27,
    'I-LAW': 28,
    'B-LOCATION': 29,
    'I-LOCATION': 30,
    'B-MONEY': 31,
    'I-MONEY': 32,
    'B-NATIONALITY': 33,
    'I-NATIONALITY': 34,
    'B-NUMBER': 35,
    'I-NUMBER': 36,
    'B-ORDINAL': 37,
    'I-ORDINAL': 38,
    'B-ORGANIZATION': 39,
    'I-ORGANIZATION': 40,
    'B-PENALTY': 41,
    'I-PENALTY': 42,
    'B-PERCENT': 43,
    'I-PERCENT': 44,
    'B-PERSON': 45,
    'I-PERSON': 46,
    'B-PRODUCT': 47,
    'I-PRODUCT': 48,
    'B-PROFESSION': 49,
    'I-PROFESSION': 50,
    'B-RELIGION': 51,
    'I-RELIGION': 52,
    'B-STATE_OR_PROVINCE': 53,
    'I-STATE_OR_PROVINCE': 54,
    'B-TIME': 55,
    'I-TIME': 56,
    'B-WORK_OF_ART': 57,
    'I-WORK_OF_ART': 58,
}

id2label = {y: x for x, y in label2id.items()}


class MyConversionScript():
    _ARCHITECTURE_TYPE_DICT = {}
    _ARCHITECTURE_TYPE_DICT = {**{"LSG" + k: v for k, v in _ARCHITECTURE_TYPE_DICT.items()}, **_ARCHITECTURE_TYPE_DICT}
    _BASE_ARCHITECTURE_TYPE = None
    _DEFAULT_ARCHITECTURE_TYPE = None
    _CONFIG_MODULE = None

    _DEFAULT_CONFIG_POSITIONAL_OFFSET = 0
    _DEFAULT_POSITIONAL_OFFSET = 0

    def __init__(
            self,
            initial_model,
            model_name,
            max_sequence_length,
            architecture,
            random_global_init,
            global_positional_stride,
            keep_first_global_token,
            resize_lsg,
            model_kwargs,
            use_token_ids,
            use_auth_token,
            config,
            save_model,
            seed
    ):

        self.initial_model = initial_model
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.architecture = architecture
        self.random_global_init = random_global_init
        self.global_positional_stride = global_positional_stride
        self.keep_first_global_token = keep_first_global_token
        self.resize_lsg = resize_lsg
        self.model_kwargs = model_kwargs
        self.use_token_ids = use_token_ids
        self.use_auth_token = use_auth_token
        self.config = config
        self.save_model = save_model

        self.new_config = None

    def save(self, model, tokenizer):

        model.save_pretrained(self.model_name)
        tokenizer.save_pretrained(self.model_name)

    def process(self):

        (lsg_architecture, lsg_model), initial_architecture = self.get_architecture()
        is_base_architecture, is_lsg, keep_first_global = self.get_additional_params(lsg_architecture,
                                                                                     initial_architecture)
        model, tokenizer = self.get_model(lsg_architecture, lsg_model)
        is_training = model.training
        model, tokenizer = self.update_config(model, tokenizer)

        # Get the module prefix to update
        module_prefix = self.get_module(model, is_base_architecture)

        # Update global embedding
        if not (is_lsg and self.resize_lsg):
            bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
            bos_id = bos_id if bos_id is not None else model.config.bos_token_id
            mask_id = tokenizer.mask_token_id
            if self.random_global_init:
                self.update_global_randomly(module_prefix, bos_id, self.global_positional_stride, keep_first_global)
            else:
                self.update_global(module_prefix, bos_id, mask_id, self.global_positional_stride, keep_first_global)

        # Update positional
        self.update_positions(module_prefix, self.max_sequence_length)

        # For Pegasus
        self.update_positions_with_model(model, self.max_sequence_length)

        if self.save_model:
            self.save(model, tokenizer)

        return model.train() if is_training else model.eval(), tokenizer

    def get_architecture(self):
        if self.architecture is not None:
            return self.validate_architecture(self.architecture)

        architectures = self.config.architectures
        if architectures is not None:
            architecture = architectures if isinstance(architectures, str) else architectures[0]
            return self.validate_architecture(architecture)

        return self.validate_architecture(self._DEFAULT_ARCHITECTURE_TYPE)

    def validate_architecture(self, architecture):
        _architecture = self._ARCHITECTURE_TYPE_DICT.get(architecture, None)

        s = "\n * " + "\n * ".join([k for k in self._ARCHITECTURE_TYPE_DICT.keys()])
        assert _architecture is not None, f"Provided/config architecture is wrong, make sure it is in: {s}"
        return _architecture, architecture

    def get_model(self, lsg_architecture, lsg_model):
        self.new_config = self._CONFIG_MODULE.from_pretrained(
            self.initial_model,
            architectures=lsg_architecture,
            trust_remote_code=True,
            use_auth_token=self.use_auth_token,
            **json.loads(self.model_kwargs.replace("'", "\""))
        )
        self.new_config.label2id = label2id
        self.new_config.id2label = id2label
        self.new_config._num_labels = len(label2id)
        model = lsg_model.from_pretrained(self.initial_model, use_auth_token=self.use_auth_token,
                                          config=self.new_config, trust_remote_code=True, ignore_mismatched_sizes=True)
        tokenizer = AutoTokenizer.from_pretrained(self.initial_model, use_auth_token=self.use_auth_token,
                                                  trust_remote_code=True, truncation=True, padding='max_length',
                                                  max_length=4096)
        return model, tokenizer

    def update_config(self, model, tokenizer):

        # Update tokenizer and config
        tokenizer.model_max_length = self.max_sequence_length
        tokenizer.init_kwargs['model_max_length'] = self.max_sequence_length

        max_pos = self.max_sequence_length
        model.config.max_position_embeddings = max_pos + self._DEFAULT_CONFIG_POSITIONAL_OFFSET
        model.config._name_or_path = self.model_name
        return model, tokenizer

    def get_additional_params(self, _architecture, initial_architecture):

        # Hack because of architecture
        is_base_architecture = True if _architecture in [self._BASE_ARCHITECTURE_TYPE,
                                                         "LSG" + self._BASE_ARCHITECTURE_TYPE] else False

        # Check if it is LSG architecture
        if vars(self.config).get("base_model_prefix", None) == "lsg" or "LSG" in initial_architecture:
            is_lsg_architecture = True
        else:
            is_lsg_architecture = False

        if is_lsg_architecture and not self.resize_lsg:
            warnings.warn(
                "LSG architecture detected, to resize positional embedding only, add --resize_lsg (won't affect global embedding)")
        if is_lsg_architecture and not self.keep_first_global_token and not self.resize_lsg:
            warnings.warn(
                "LSG architecture detected, to keep the same first global token, add --keep_first_global_token")

        keep_first = False
        if self.keep_first_global_token:
            if is_lsg_architecture:
                keep_first = True
            else:
                warnings.warn("--keep_first_global_token won't be used if the initial model isn't a LSG model")
        return is_base_architecture, is_lsg_architecture, keep_first

    def get_module(self, model, is_base_architecture):
        if is_base_architecture:
            return
        return

    def update_global_randomly(self, module_prefix, bos_id, stride, keep_first_global):
        pass

    def update_global(self, module_prefix, bos_id, mask_id, stride, keep_first_global):
        pass

    def update_positions(self, module_prefix, max_pos):
        pass

    def update_positions_with_model(self, model, max_pos):
        pass

    def update_buffer(self, module, value):
        pass

    def order_positions(self, positions, stride):
        n, d = positions.size()
        if n % 512 != 0:
            if n > 512:
                positions = positions[:512 * (n // 512)]
            else:
                mean = positions.mean(dim=0, keepdim=True).expand(512 - n, -1)
                std = positions.std(dim=0, keepdim=True).expand(512 - n, -1)
                positions = torch.cat([positions, torch.normal(mean, std)], dim=0)
            n, d = positions.size()

        factor = n // 512
        positions = positions.reshape(-1, factor, d)[:, 0]
        positions = positions.reshape(-1, stride // factor, d).transpose(0, 1).reshape(-1, d)
        return positions

    def run_test(self):
        pass

    def run_models(self, lsg_path, max_length, hidden_size, text, auto_map, gradient_checkpointing=False,
                   is_encoder_decoder=False):

        from transformers import AutoTokenizer, AutoConfig, AutoModel, pipeline
        from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, \
            AutoModelForQuestionAnswering
        from transformers import AutoModelForMaskedLM, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(lsg_path)

        long_text = text * 200
        dtype = torch.bfloat16

        for name in auto_map.keys():

            if name == "AutoConfig":
                continue

            model = getattr(sys.modules["transformers"], name)
            print("\n\n" + "=" * 5 + " " + name + " " + "=" * 5 + "\n")
            model = model.from_pretrained(lsg_path, trust_remote_code=True, is_decoder="Causal" in name,
                                          torch_dtype=dtype).train()

            if gradient_checkpointing:
                model.gradient_checkpointing_enable()

            if "QuestionAnswering" in name:
                tokens = tokenizer("context", long_text, return_tensors="pt", truncation=True)
                inputs_embeds = torch.randn(1, max_length, hidden_size, dtype=dtype)
            elif "MultipleChoice" in name:
                num_choices = 4
                tokens = tokenizer([long_text] * num_choices, return_tensors="pt", truncation=True)
                tokens = {k: v.reshape(1, num_choices, -1) for k, v in tokens.items()}
                inputs_embeds = torch.randn(1, num_choices, max_length // 4, hidden_size, dtype=dtype)
            else:
                tokens = tokenizer(long_text, return_tensors="pt", truncation=True)
                inputs_embeds = torch.randn(1, max_length, hidden_size, dtype=dtype)

            if model.config.model_type != "pegasus":
                model(**tokens)

            if not is_encoder_decoder:
                model(inputs_embeds=inputs_embeds)
            elif "decoder_input_ids" in model.forward.__code__.co_varnames:
                decoder_input_ids = tokens.input_ids[:, :256]
                if "SequenceClassification" not in name:
                    model(**tokens, decoder_input_ids=decoder_input_ids)


from lsg_converter.bert.modeling_lsg_bert import *
from lsg_converter.conversion_utils import ConversionScript


class MyBertConversionScript(MyConversionScript):
    _ARCHITECTURE_TYPE_DICT = {
        "BertModel": ("LSGBertModel", LSGBertModel),
        "BertForMaskedLM": ("LSGBertForMaskedLM", LSGBertForMaskedLM),
        "BertForPreTraining": ("LSGBertForPreTraining", LSGBertForPreTraining),
        "BertLMHeadModel": ("LSGBertLMHeadModel", LSGBertLMHeadModel),
        "BertForMultipleChoice": ("LSGBertForMultipleChoice", LSGBertForMultipleChoice),
        "BertForQuestionAnswering": ("LSGBertForQuestionAnswering", LSGBertForQuestionAnswering),
        "BertForSequenceClassification": ("LSGBertForSequenceClassification", LSGBertForSequenceClassification),
        "BertForTokenClassification": ("LSGBertForTokenClassification", LSGBertForTokenClassification)
    }
    _ARCHITECTURE_TYPE_DICT = {**{"LSG" + k: v for k, v in _ARCHITECTURE_TYPE_DICT.items()}, **_ARCHITECTURE_TYPE_DICT}

    _BASE_ARCHITECTURE_TYPE = "BertModel"
    _DEFAULT_ARCHITECTURE_TYPE = "BertForPreTraining"
    _CONFIG_MODULE = LSGBertConfig

    _DEFAULT_CONFIG_POSITIONAL_OFFSET = 0
    _DEFAULT_POSITIONAL_OFFSET = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_module(self, model, is_base_architecture):
        if is_base_architecture:
            return model
        return model.bert

    def update_global_randomly(self, module_prefix, bos_id, stride, keep_first_global):

        import torch
        from torch.distributions.multivariate_normal import MultivariateNormal

        u = module_prefix.embeddings.word_embeddings.weight.clone()

        cov = torch.cov(u.T)
        m = MultivariateNormal(u.mean(dim=0), cov)
        w = m.sample((512,))
        w[0] = u[bos_id]

        positions = module_prefix.embeddings.position_embeddings.weight.clone()
        positions = self.order_positions(positions, stride)

        if self.use_token_ids:
            token_ids = module_prefix.embeddings.token_type_embeddings.weight.clone()
            positions += token_ids[0].unsqueeze(0)
            w[0] = u[bos_id] + token_ids[0]

        if keep_first_global:
            module_prefix.embeddings.global_embeddings.weight.data[1:] = (w + positions)[1:]
        else:
            module_prefix.embeddings.global_embeddings.weight.data = w + positions

    def update_global(self, module_prefix, bos_id, mask_id, stride, keep_first_global):

        u = module_prefix.embeddings.word_embeddings.weight.clone()
        positions = module_prefix.embeddings.position_embeddings.weight.clone()
        positions = self.order_positions(positions, stride)

        positions[0] += u[bos_id]
        positions[1:] += u[mask_id].unsqueeze(0)

        if self.use_token_ids:
            token_ids = module_prefix.embeddings.token_type_embeddings.weight.clone()
            positions += token_ids[0].unsqueeze(0)

        if keep_first_global:
            module_prefix.embeddings.global_embeddings.weight.data[1:] = positions[1:]
        else:
            module_prefix.embeddings.global_embeddings.weight.data = positions

    def update_positions(self, module_prefix, max_pos):

        position_embeddings_weights = module_prefix.embeddings.position_embeddings.weight.clone()
        current_max_position = position_embeddings_weights.size()[0]

        new_position_embeddings_weights = torch.cat([
            position_embeddings_weights for _ in range(max_pos // current_max_position + 1)
        ], dim=0)[:max_pos + self._DEFAULT_POSITIONAL_OFFSET]

        module_prefix.embeddings.position_embeddings = nn.Embedding(
            *new_position_embeddings_weights.size(),
            _weight=new_position_embeddings_weights,
            dtype=new_position_embeddings_weights.dtype
        )
        self.update_buffer(module_prefix.embeddings, max_pos + self._DEFAULT_POSITIONAL_OFFSET)

    def update_buffer(self, module, value):

        # Update buffer dogshit
        module.register_buffer(
            "position_ids", torch.arange(value).expand((1, -1)), persistent=False
        )
        module.register_buffer(
            "token_type_ids", torch.zeros(module.position_ids.size(), dtype=torch.long), persistent=False
        )

    def run_test(self):

        from transformers import AutoConfig, AutoTokenizer

        initial_path = self.initial_model
        lsg_path = self.model_name

        config = AutoConfig.from_pretrained(lsg_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(lsg_path)
        text = f"Paris is the {tokenizer.mask_token} of France."

        max_length = config.max_position_embeddings - 20
        hidden_size = config.hidden_size

        self.run_models(lsg_path, max_length, hidden_size, text, AUTO_MAP)
        self.run_pipeline(lsg_path, initial_path, tokenizer, text)

    def run_pipeline(self, lsg_path, initial_path, tokenizer, text):

        from transformers import AutoModelForMaskedLM, pipeline

        model = AutoModelForMaskedLM.from_pretrained(lsg_path, trust_remote_code=True)
        pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        pipe_lsg = pipe(text)

        model = AutoModelForMaskedLM.from_pretrained(initial_path, trust_remote_code=True)
        pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        pipe_initial = pipe(text)

        print("\n\n" + "=" * 5 + " LSG PIPELINE " + "=" * 5 + "\n")
        print(text)
        print(pipe_lsg[0])
        print("\n\n" + "=" * 5 + " INITIAL PIPELINE " + "=" * 5 + "\n")
        print(text)
        print(pipe_initial[0])


from transformers import AutoConfig
from transformers.models.auto.modeling_auto import *
import json

from lsg_converter.albert.convert_albert_checkpoint import *
from lsg_converter.bart.convert_bart_checkpoint import *
from lsg_converter.barthez.convert_barthez_checkpoint import *
from lsg_converter.bert.convert_bert_checkpoint import *
from lsg_converter.camembert.convert_camembert_checkpoint import *
from lsg_converter.distilbert.convert_distilbert_checkpoint import *
from lsg_converter.electra.convert_electra_checkpoint import *
from lsg_converter.mbart.convert_mbart_checkpoint import *
from lsg_converter.pegasus.convert_pegasus_checkpoint import *
from lsg_converter.roberta.convert_roberta_checkpoint import *
from lsg_converter.xlm_roberta.convert_xlm_roberta_checkpoint import *

_AUTH_MODELS = {
    "albert": AlbertConversionScript,
    "bart": BartConversionScript,
    "barthez": BarthezConversionScript,
    "bert": MyBertConversionScript,
    "camembert": CamembertConversionScript,
    "distilbert": DistilBertConversionScript,
    "electra": ElectraConversionScript,
    "mbart": MBartConversionScript,
    "pegasus": PegasusConversionScript,
    "roberta": RobertaConversionScript,
    "xlm-roberta": XLMRobertaConversionScript,
}


class MYLSGConverter():

    def __init__(
            self,
            max_sequence_length=4096,
            random_global_init=False,
            global_positional_stride=64,
            keep_first_global_token=False,
            resize_lsg=False,
            use_token_ids=True,
            seed=123
    ):
        """
        max_sequence_length (int): new max sequence length
        random_global_init (bool): randomly initialize global tokens
        global_positional_stride (int): position stride between global tokens
        keep_first_global_token (bool): keep or replace the first global token (<s> + pos 0)
        resize_lsg (bool): only resize an existing LSG model
        use_token_ids (bool): use token_type_ids to build global tokens
        seed (int): seed
        """
        self.max_sequence_length = max_sequence_length
        self.random_global_init = random_global_init
        self.global_positional_stride = global_positional_stride
        self.keep_first_global_token = keep_first_global_token
        self.resize_lsg = resize_lsg
        self.use_token_ids = use_token_ids
        self.seed = seed

    def convert_from_pretrained(
            self,
            model_name_or_path,
            architecture=None,
            use_auth_token=False,
            **model_kwargs
    ):
        """
        mode_name_or_path (str): path to the model to convert
        architecture (str): specific architecture (optional)
        model_kwargs: additional model args
        """

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, use_auth_token=use_auth_token)
        config.label2id = label2id
        config.id2label = id2label
        config._num_labels = len(label2id)

        model_type = config.model_type
        model_kwargs = json.dumps(model_kwargs, indent=4)

        if model_type in _AUTH_MODELS.keys():
            converter = _AUTH_MODELS[model_type](
                initial_model=model_name_or_path,
                model_name=model_name_or_path,
                max_sequence_length=self.max_sequence_length,
                architecture=architecture,
                random_global_init=self.random_global_init,
                global_positional_stride=self.global_positional_stride,
                keep_first_global_token=self.keep_first_global_token,
                resize_lsg=self.resize_lsg,
                model_kwargs=model_kwargs,
                use_token_ids=self.use_token_ids,
                use_auth_token=use_auth_token,
                config=config,
                save_model=False,
                seed=self.seed
            )

            return converter.process()