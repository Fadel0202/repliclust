from abc import ABC, abstractmethod
import json
import os
import requests
from typing import Optional, Dict, Any, Union

class LLMClient(ABC):
    """Interface abstraite pour les clients de modèles de langage."""
    
    @abstractmethod
    def get_completion(self, system_prompt: str, user_prompt: str, 
                      temperature: float = 0.0, 
                      response_format: Optional[Dict[str, str]] = None, 
                      seed: Optional[int] = None) -> str:
        """Obtenir une complétion du modèle de langage."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Vérifier si le client est correctement initialisé et disponible."""
        pass


class OpenAIClient(LLMClient):
    """Client OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialiser le client OpenAI."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self._available = True
        except Exception as e:
            self.client = None
            self._available = False
            print(f"OpenAI non disponible: {str(e)}")
    
    def get_completion(self, system_prompt: str, user_prompt: str, 
                      temperature: float = 0.0, 
                      response_format: Optional[Dict[str, str]] = None, 
                      seed: Optional[int] = None) -> str:
        if not self._available:
            raise ValueError("Client OpenAI non disponible")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        if seed is not None:
            kwargs["seed"] = seed
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def is_available(self) -> bool:
        return self._available


class LlamaClient(LLMClient):
    """Client pour l'API LLaMA."""
    
    def __init__(self, api_url: str = "http://localhost:8080", model: Optional[str] = None):
        """
        Initialiser le client LLaMA.
        
        Parameters
        ----------
        api_url : str
            URL de l'endpoint API LLaMA (par défaut: "http://localhost:8080")
        model : str, optional
            Modèle LLaMA spécifique à utiliser si l'API en supporte plusieurs
        """
        self.api_url = api_url
        self.model = model
        
        # Vérifier si l'API est disponible
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            self._available = response.status_code == 200
        except Exception:
            self._available = False
            print(f"LLaMA API non disponible à {api_url}")
    
    def get_completion(self, system_prompt: str, user_prompt: str, 
                      temperature: float = 0.0, 
                      response_format: Optional[Dict[str, str]] = None, 
                      seed: Optional[int] = None) -> str:
        if not self._available:
            raise ValueError("API LLaMA non disponible")
        
        # Formater le prompt selon le format attendu par LLaMA
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        
        if self.model:
            data["model"] = self.model
            
        if seed is not None:
            data["seed"] = seed
        
        # Ajouter une indication pour formater en JSON si nécessaire
        if response_format and response_format.get("type") == "json_object":
            data["prompt"] += " Réponds uniquement avec un objet JSON valide."
        
        response = requests.post(f"{self.api_url}/completion", json=data)
        
        if response.status_code != 200:
            raise Exception(f"Erreur API LLaMA: {response.text}")
        
        result = response.json()
        response_text = result.get("content", "")
        
        # Si JSON demandé, s'assurer qu'il est valide
        if response_format and response_format.get("type") == "json_object":
            try:
                # Trouver le JSON dans la réponse s'il est entouré d'autre texte
                import re
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # Valider le JSON en l'analysant et le re-sérialisant
                    parsed_json = json.loads(json_str)
                    return json.dumps(parsed_json)
                else:
                    # Si aucun objet JSON trouvé, essayer d'analyser toute la réponse
                    json.loads(response_text)
                    return response_text
            except json.JSONDecodeError:
                # Si on ne peut pas analyser le JSON, retourner la réponse brute
                return response_text
        
        return response_text
    
    def is_available(self) -> bool:
        return self._available


def get_llm_client(api_key: Optional[str] = None, use_llama: bool = False,
                  llama_url: str = "http://localhost:8080") -> LLMClient:
    """
    Obtenir le meilleur client LLM disponible.
    
    Parameters
    ----------
    api_key : str, optional
        Clé API pour OpenAI
    use_llama : bool, optional
        Si True, utiliser LLaMA même si OpenAI est disponible
    llama_url : str, optional
        URL de l'API LLaMA
        
    Returns
    -------
    LLMClient
        Un client LLM initialisé et disponible
        
    Raises
    ------
    Exception
        Si aucun client LLM n'a pu être initialisé
    """
    if use_llama:
        # Utiliser LLaMA directement
        llama_client = LlamaClient(api_url=llama_url)
        if llama_client.is_available():
            return llama_client
        raise Exception("Client LLaMA demandé mais non disponible")
    
    # Essayer OpenAI en premier
    if api_key or os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAIClient(api_key=api_key)
        if openai_client.is_available():
            return openai_client
    
    # Repli sur LLaMA
    llama_client = LlamaClient(api_url=llama_url)
    if llama_client.is_available():
        return llama_client
    
    # Si nous arrivons ici, aucun client n'est disponible
    raise Exception(
        "Aucun client LLM n'a pu être initialisé. "
        "Veuillez fournir une clé API OpenAI ou vous assurer que l'API LLaMA est en cours d'exécution."
    )